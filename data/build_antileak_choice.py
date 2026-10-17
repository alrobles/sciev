#!/usr/bin/env python3
"""Anti-leakage training controls for choice decisions.

Unlike noul/score, a corrupted-evidence choice row has no reject class:
the criteria are N candidate answers and one must win. Two supervision
signals replace the reject remap:

  empty      -> soft = [1/N]*N embedded in the row (KL-to-uniform when
                trained with --soft-weight 1.0): "no usable evidence ->
                no preference".
  transplant -> the record's evidence span is replaced by another
                record's evidence while the question/options are kept,
                and a *noul teacher* (the anti-leak alp25 head) verifies
                every option as a "Proposed answer" claim against the
                donor passage. Confident support -> hard gold at the
                supported option (evidence-following, possibly
                counterfactual); no confident support -> uniform soft.

Donors are ranked by idf-weighted overlap with the question + option
texts (--donor near) so transplants are topical near-misses — the
confounder a retrieval system actually faces — rather than random
foreign passages.

Two passes (the teacher runs as a normal sciev.eval GPU job):

    python data/build_antileak_choice.py emit \
        --text sci_decisions_train_text.jsonl --control-frac 0.25 \
        --controls-out empty_controls.jsonl \
        --candidates transplant_candidates.jsonl \
        --teacher-out teacher_noul_reqs.jsonl
    python -m sciev.eval --ckpt runs/sci/c_noul_alp25/decision.pt \
        --decision-type noul --decisions-eval teacher_noul_reqs.jsonl \
        --predictions --out teacher_eval.json
    python data/build_antileak_choice.py relabel \
        --candidates transplant_candidates.jsonl \
        --teacher-eval teacher_eval.json \
        --controls-out transplant_controls.jsonl \
        --audit transplant_audit.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from sciev.data import (_evidence_source, _mark_evidence_control,
                        add_decision, exclude_record, input_fingerprints,
                        make_evidence_controls, read_jsonl, write_dataset)

MAX_CTX, MAX_OPT = 960, 120
NOUL_INSTRUCTIONS = ("Does the proposed answer correctly answer the "
                     "question according to the supplied passage?")
STOPWORDS = frozenset(
    "the a an and or of to in on for with by is are was were be been "
    "that this these those it its as at from into over under between "
    "which what who whom whose when where why how not no than then "
    "such more most other some any each all both each can could may "
    "might will would shall should do does did have has had about "
    "their there here also but if so we you they he she i".split())
TERM_RE = re.compile(r"[a-z0-9]{4,}")


def _terms(text):
    return Counter(t for t in TERM_RE.findall(text.lower())
                   if t not in STOPWORDS)


def _choice_units(text_rows):
    """(record, qid, question) units for choice questions only."""
    units = []
    for record in text_rows:
        for qid, question in record.get("questions", {}).items():
            if question.get("type") == "choice":
                units.append((record, qid, question))
    return units


def _gold_index(question):
    """Choice labels are option *texts*; gold index is position in criteria."""
    criteria = question.get("criteria")
    label = question.get("label")
    if not isinstance(criteria, dict) or label not in criteria:
        return None
    return list(criteria).index(label)


def _idf_index(doc_terms):
    df = Counter()
    for terms in doc_terms:
        df.update(set(terms))
    n_docs = max(1, len(doc_terms))
    return {term: math.log(1 + n_docs / count) for term, count in df.items()}


def _rank_donor(query_terms, doc_terms, idf, eligible, rng):
    best, best_score = None, 0.0
    for index in eligible:
        score = sum(idf.get(term, 0.0) * doc_terms[index][term]
                    for term in query_terms if term in doc_terms[index])
        if score > best_score:
            best, best_score = index, score
    if best is None and eligible:
        best = rng.choice(eligible)
    return best


def _question_tail(state):
    marker = "\nQuestion:"
    return state.rsplit(marker, 1)[1].strip() if marker in state else ""


def build_empty_controls(pool, tok, *, control_frac, seed, max_ctx, max_opt,
                         overflow, exclusions):
    """Evidence-removed rows carrying a uniform soft target."""
    id_rows, soft_rows, text_by_state = [], [], {}
    records = make_evidence_controls(pool, mode="empty", seed=seed)
    units = []
    for record in records:
        for qid, question in record["questions"].items():
            if question.get("type") == "choice":
                units.append((record, qid, question))
    keep = max(1, round(len(units) * control_frac))
    rng = random.Random(seed)
    for record, qid, question in (units[i] for i in
                                  sorted(rng.sample(range(len(units)), keep))):
        n_opts = len(question.get("criteria") or {})
        if n_opts < 2:
            exclude_record(exclusions, record, "choice_criteria_missing", qid=qid)
            continue
        meta = {key: val for key, val in record.items()
                if key not in ("state", "questions")}
        meta["gold_semantics"] = "uniform_target"
        meta["label_status"] = "antileak_train"
        question = dict(question)
        question["label_status"] = "antileak_train"
        question["gold_semantics"] = "uniform_target"
        before = len(id_rows)
        add_decision(id_rows, text_by_state, tok, record["state"], qid,
                     question, 0, meta, max_ctx, max_opt, overflow, exclusions)
        if len(id_rows) > before:
            id_rows[-1]["soft"] = [1.0 / n_opts] * n_opts
            soft_rows.append({"qid": id_rows[-1]["qid"],
                              "soft": id_rows[-1]["soft"],
                              "mode": "empty"})
    return id_rows, soft_rows, list(text_by_state.values())


def _perturb_edits(question, evidence):
    """Provenance distractors that differ from the gold claim by a single
    token span -> (option_index, src, dst) when src appears verbatim in
    the evidence. Editing src -> dst inside the passage yields a minimal
    counterfactual: same question, same passage, different supported
    answer."""
    from difflib import SequenceMatcher
    criteria = question.get("criteria") or {}
    gold_key = question.get("label")
    if not isinstance(gold_key, str) or gold_key not in criteria:
        return []
    out = []
    gold_tokens = gold_key.split()
    for prov in question.get("negative_provenance") or []:
        distr = prov.get("option_key")
        if not isinstance(distr, str) or distr not in criteria or distr == gold_key:
            continue
        ops = [op for op in
               SequenceMatcher(None, gold_tokens, distr.split()).get_opcodes()
               if op[0] != "equal"]
        if len(ops) != 1 or ops[0][0] != "replace":
            continue
        _, a0, a1, b0, b1 = ops[0]
        src, dst = " ".join(gold_tokens[a0:a1]), " ".join(distr.split()[b0:b1])
        if src and dst and re.search(
                r"(?<![\w.])" + re.escape(src) + r"(?![\w.])", evidence):
            out.append((list(criteria).index(distr), src, dst, prov))
    return out


def _provenance_donors(question, by_sample, by_pid):
    """cross_passage provenance entries -> (option_index, donor_index).

    A distractor's provenance records the foreign passage its claim was
    drawn from — splicing that passage produces a true counterfactual:
    evidence that supports a different option.
    """
    out = []
    criteria = question.get("criteria") or {}
    for prov in question.get("negative_provenance") or []:
        if prov.get("strategy") != "cross_passage":
            continue
        option_key = prov.get("option_key")
        if option_key not in criteria:
            continue
        donor_index = None
        sample_id = prov.get("source_sample_id")
        if sample_id is not None and sample_id in by_sample:
            donor_index = by_sample[sample_id]
        else:
            donor_index = by_pid.get(str(prov.get("source_pid")))
        if donor_index is not None:
            out.append((list(criteria).index(option_key), donor_index, prov))
    return out


def build_transplant_candidates(units, pool, tok, *, control_frac, seed,
                                donor, max_ctx, max_opt, overflow, exclusions):
    """Evidence-edited rows + per-option noul teacher requests.

    donor is a comma-separated set of mechanisms:
      provenance -> splice the cross_passage source passage (distractor's
                    claim origin)
      perturb    -> apply the distractor's single-token edit inside the
                    record's own evidence (minimal counterfactual)
      near|random-> splice an idf-ranked or random different-group passage
    """
    modes = {m.strip() for m in re.split(r"[,+]", donor) if m.strip()}
    rng = random.Random(seed)
    sources = [_evidence_source(record, index)
               for index, record in enumerate(pool)]
    doc_terms = [_terms(evidence) for _, _, evidence in sources]
    idf = _idf_index(doc_terms)
    record_index = {id(record): i for i, record in enumerate(pool)}
    by_sample = {r.get("sample_id"): i for i, r in enumerate(pool)
                 if r.get("sample_id")}
    by_pid = {str(r.get("pid")): i for i, r in enumerate(pool)
              if r.get("pid") is not None}

    # work item: (record, qid, question, donor_index, intended_gold,
    #             prov, edited_evidence)
    work = []
    for record, qid, question in units:
        index = record_index[id(record)]
        _, _, evidence = sources[index]
        if "provenance" in modes:
            for opt_index, donor_index, prov in _provenance_donors(
                    question, by_sample, by_pid):
                work.append((record, qid, question, donor_index,
                             opt_index, prov, None))
        if "perturb" in modes:
            for opt_index, src, dst, prov in _perturb_edits(question,
                                                            evidence):
                work.append((record, qid, question, None, opt_index, prov,
                             (src, dst)))
    ranked = modes & {"near", "random"}
    if ranked:
        keep = max(1, round(len(units) * control_frac))
        for u in (units[i] for i in
                  sorted(rng.sample(range(len(units)), keep))):
            work.append((u[0], u[1], u[2], None, None, None, None))

    candidates, teacher_rows, teacher_text = [], [], {}
    seen = set()
    for record, qid, question, forced_donor, intended_gold, prov, edit in work:
        index = record_index[id(record)]
        group, span, evidence = sources[index]
        n_opts = len(question.get("criteria") or {})
        gold = _gold_index(question)
        if n_opts < 2 or gold is None:
            exclude_record(exclusions, record, "choice_label_not_in_criteria",
                           qid=qid)
            continue
        if edit is not None:
            src, dst = edit
            new_evidence = re.sub(
                r"(?<![\w.])" + re.escape(src) + r"(?![\w.])",
                lambda _: dst, evidence)
            if new_evidence == evidence:
                exclude_record(exclusions, record, "perturb_edit_no_effect",
                               qid=qid)
                continue
            donor_index, donor_evidence, donor_group = None, new_evidence, group
            donor_kind = "perturb"
        elif forced_donor is not None:
            donor_index = forced_donor
            if (donor_index == index or sources[donor_index][0] == group
                    or sources[donor_index][2] == evidence):
                exclude_record(exclusions, record, "self_donor", qid=qid)
                continue
            donor_group, _, donor_evidence = sources[donor_index]
            donor_kind = "provenance"
        else:
            donor_kind = sorted(ranked)[0]
            eligible = [i for i, (g, _, other) in enumerate(sources)
                        if i != index and g != group and other != evidence]
            if not eligible:
                exclude_record(exclusions, record, "no_transplant_donor",
                               qid=qid)
                continue
            query = _question_tail(record["state"]) + " " + " ".join(
                question["criteria"])
            if donor_kind == "near":
                donor_index = _rank_donor(_terms(query), doc_terms, idf,
                                          eligible, rng)
            else:
                donor_index = rng.choice(eligible)
            donor_group, _, donor_evidence = sources[donor_index]
        controlled = deepcopy(record)
        start, end = span
        controlled["state"] = (record["state"][:start] + donor_evidence
                               + record["state"][end:])
        new_span = (start, start + len(donor_evidence))
        state_digest = hashlib.sha256(
            controlled["state"].encode()).hexdigest()[:16]
        if (qid, state_digest) in seen:
            continue
        seen.add((qid, state_digest))
        marker = {"mode": "transplant", "donor": donor_kind, "seed": seed,
                  "source_group": group, "donor_group": donor_group,
                  "source_record_index": index, "donor_record_index": donor_index,
                  "label_semantics": "teacher_relabelled"}
        _mark_evidence_control(controlled, marker, new_span)
        identity = {"state": controlled["state"], "mode": "transplant",
                    "source_group": group, "source_record_index": index,
                    "donor_record_index": donor_index,
                    "reference_qid": qid,
                    "question": {key: question[key]
                                 for key in ("type", "instructions", "criteria")
                                 if key in question}}
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                                           allow_nan=False).encode()).hexdigest()
        cand_qid = f"{qid}__evidence_transplant_{digest[:16]}"
        question_c = dict(question)
        _mark_evidence_control(question_c, marker, new_span)
        question_c.update(qid=cand_qid, reference_qid=qid,
                          label_status="antileak_train")
        controlled["questions"] = {cand_qid: question_c}

        meta = {key: val for key, val in controlled.items()
                if key not in ("state", "questions")}
        meta["gold_semantics"] = "teacher_relabelled"
        meta["label_status"] = "antileak_train"
        id_rows = []
        before = len(id_rows)
        add_decision(id_rows, {}, tok, controlled["state"], cand_qid,
                     question_c, gold, meta, max_ctx, max_opt, overflow,
                     exclusions)
        if len(id_rows) == before:
            continue
        encoded = id_rows[0]
        option_keys = encoded.get("option_keys") or list(question["criteria"])

        teacher_qids = []
        ok = True
        for opt_index, option_text in enumerate(option_keys):
            t_state = (controlled["state"]
                       + f"\nProposed answer: {option_text}")
            t_qid = f"{cand_qid}__opt{opt_index}"
            t_question = {"type": "noul", "instructions": NOUL_INSTRUCTIONS,
                          "criteria": None, "label_status": "teacher_request"}
            t_meta = {"label_status": "teacher_request",
                      "candidate_qid": cand_qid, "option_index": opt_index,
                      "option_text": option_text,
                      "source": "build_antileak_choice"}
            before_t = len(teacher_rows)
            add_decision(teacher_rows, teacher_text, tok, t_state, t_qid,
                         t_question, 0, t_meta, max_ctx, max_opt, overflow,
                         exclusions)
            if len(teacher_rows) == before_t:
                ok = False
                break
            teacher_qids.append(teacher_rows[-1]["qid"])
        if not ok or len(teacher_qids) != len(option_keys):
            exclude_record(exclusions, meta, "teacher_request_failed",
                           qid=cand_qid)
            teacher_rows = [row for row in teacher_rows
                            if row.get("candidate_qid") != cand_qid]
            continue
        candidates.append({"candidate_qid": cand_qid,
                           "reference_qid": qid,
                           "encoded_row": encoded,
                           "option_keys": option_keys,
                           "original_gold": gold,
                           "intended_gold": intended_gold,
                           "teacher_qids": teacher_qids,
                           "donor": donor_kind,
                           "source_group": group,
                           "donor_group": donor_group,
                           "sample_id": record.get("sample_id"),
                           "split_group": record.get("split_group")})
    return candidates, teacher_rows, list(teacher_text.values())


def relabel(candidates, predictions, *, tau, margin):
    """Join teacher P(true) per option -> hard gold or uniform soft."""
    probs_by_qid = {rec["qid"]: rec["probabilities"] for rec in predictions}
    out_rows, audit, counts = [], [], Counter()
    for cand in candidates:
        p_true, missing = [], False
        for t_qid in cand["teacher_qids"]:
            probs = probs_by_qid.get(t_qid)
            if probs is None:
                missing = True
                break
            p_true.append(float(probs[0]))  # keys [true,false]
        row = dict(cand["encoded_row"])
        n_opts = len(cand["option_keys"])
        if missing:
            decision = "teacher_missing"
            counts[decision] += 1
            audit.append({**cand, "encoded_row": None, "decision": decision,
                          "p_true": p_true})
            continue
        order = sorted(range(n_opts), key=lambda i: p_true[i], reverse=True)
        winner, second = order[0], (order[1] if n_opts > 1 else None)
        confident = (p_true[winner] >= tau and
                     (second is None or p_true[winner] - p_true[second] >= margin))
        if confident:
            row["gold"] = winner
            if winner == cand["original_gold"]:
                decision = "keep_gold"
            elif winner == cand.get("intended_gold"):
                decision = "prov_confirmed"
            else:
                decision = "relabel"
        else:
            row["soft"] = [1.0 / n_opts] * n_opts
            decision = "uniform"
        counts[decision] += 1
        audit.append({**cand, "encoded_row": None, "decision": decision,
                      "p_true": p_true, "new_gold": row["gold"],
                      "confident": confident})
        out_rows.append(row)
    return out_rows, audit, counts


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    em = sub.add_parser("emit", help="build controls + teacher requests")
    em.add_argument("--text", required=True)
    em.add_argument("--control-frac", type=float, default=0.25)
    em.add_argument("--donor", default="near",
                    help="comma- or +-separated set: near|random "
                         "(idf-ranked/random foreign passage, "
                         "--control-frac applies), provenance "
                         "(cross_passage distractor source passages, all "
                         "entries), perturb (single-token counterfactual "
                         "edits inside the passage). NOTE: use + not , "
                         "when passing through sbatch --export")
    em.add_argument("--controls-out", required=True,
                    help="encoded empty controls jsonl")
    em.add_argument("--candidates", required=True,
                    help="transplant candidates jsonl")
    em.add_argument("--teacher-out", required=True,
                    help="encoded noul teacher requests jsonl")
    em.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    em.add_argument("--seed", type=int, default=7331)
    em.add_argument("--max-ctx", type=int, default=MAX_CTX)
    em.add_argument("--max-opt", type=int, default=MAX_OPT)
    em.add_argument("--overflow", choices=("error", "exclude", "truncate"),
                    default="exclude")
    rl = sub.add_parser("relabel", help="join teacher predictions")
    rl.add_argument("--candidates", required=True)
    rl.add_argument("--teacher-eval", required=True,
                    help="sciev.eval --predictions json")
    rl.add_argument("--tau", type=float, default=0.6)
    rl.add_argument("--margin", type=float, default=0.2)
    rl.add_argument("--controls-out", required=True)
    rl.add_argument("--audit", required=True)
    args = ap.parse_args()

    if args.cmd == "emit":
        if not 0 < args.control_frac <= 1.0:
            ap.error("--control-frac must be in (0, 1]")
        modes = {m.strip() for m in re.split(r"[,+]", args.donor)
                 if m.strip()}
        if not modes or not modes <= {"near", "random", "provenance",
                                      "perturb"}:
            ap.error("--donor must be a nonempty subset of "
                     "near,random,provenance,perturb")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                            trust_remote_code=True)
        text_rows = [row for _, row in read_jsonl(args.text)]
        pool = [r for r in text_rows
                if any(q.get("type") == "choice"
                       for q in r.get("questions", {}).values())]
        if not pool:
            raise SystemExit("no records carry a choice question")
        exclusions = []
        id_rows, soft_rows, text_out = build_empty_controls(
            pool, tok, control_frac=args.control_frac, seed=args.seed,
            max_ctx=args.max_ctx, max_opt=args.max_opt,
            overflow=args.overflow, exclusions=exclusions)
        units = _choice_units(text_rows)
        candidates, teacher_rows, teacher_text = build_transplant_candidates(
            units, pool, tok, control_frac=args.control_frac,
            seed=args.seed + 1, donor=args.donor, max_ctx=args.max_ctx,
            max_opt=args.max_opt, overflow=args.overflow,
            exclusions=exclusions)
        out = Path(args.controls_out)
        manifest = {"builder": "build_antileak_choice", "builder_version": 1,
                    "phase": "emit", "modes": ["empty"],
                    "seed": args.seed, "control_frac": args.control_frac,
                    "inputs": input_fingerprints([args.text]),
                    "tokenizer": args.tokenizer,
                    "max_ctx": args.max_ctx, "max_opt": args.max_opt,
                    "overflow": args.overflow, "exclusions": exclusions,
                    "counts": {"input_records": len(text_rows),
                               "choice_units": len(units),
                               "empty_rows": len(id_rows),
                               "transplant_candidates": len(candidates),
                               "teacher_requests": len(teacher_rows)}}
        write_dataset({out: id_rows,
                       out.with_name(f"{out.stem}_text.jsonl"): text_out},
                      out.with_suffix(".manifest.json"), manifest)
        _write_jsonl(args.candidates, candidates)
        _write_jsonl(args.teacher_out, teacher_rows)
        teacher_out = Path(args.teacher_out)
        _write_jsonl(teacher_out.with_name(teacher_out.stem + "_text.jsonl"),
                     teacher_text)
        from sciev.data import versioned_path
        versioned = versioned_path(out)
        soft_path = versioned.with_name(versioned.stem + ".soft.jsonl")
        _write_jsonl(soft_path, soft_rows)
        print(f"[emit] empty={len(id_rows)} candidates={len(candidates)} "
              f"teacher={len(teacher_rows)} excluded={len(exclusions)}",
              flush=True)
    else:
        candidates = [row for _, row in read_jsonl(args.candidates)]
        report = json.loads(Path(args.teacher_eval).read_text())
        predictions = (report.get("predictions")
                       or (report.get("decisions_eval") or {}).get("predictions")
                       or [])
        out_rows, audit, counts = relabel(candidates, predictions,
                                          tau=args.tau, margin=args.margin)
        from sciev.data import versioned_path
        controls_path = versioned_path(Path(args.controls_out))
        controls_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(controls_path, out_rows)
        _write_jsonl(args.audit, audit)
        soft_rows = [{"qid": row["qid"], "soft": row["soft"],
                      "mode": "transplant"}
                     for row in out_rows if "soft" in row]
        _write_jsonl(controls_path.with_name(controls_path.stem + ".soft.jsonl"),
                     soft_rows)
        print(f"[relabel] rows={len(out_rows)} {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
