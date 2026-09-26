#!/usr/bin/env python3
"""Build K-way *scientific* decision batteries from ecoreasoner QA data.

Source: qa_devin_elite.jsonl — {pid, type, q, a, passage} where type in
{numerical, causal, multihop, negation, definitional}.

Decision tasks (same shape as the tool-call battery):
  sci_choice_{train,dev,eval}.jsonl   K=4 answers, gold correct
  sci_noul_{train,dev,eval}.jsonl     K=2, is the proposed answer supported?
  sci_score_{train,dev,eval}.jsonl    K=3 rubric: unrelated / related-but-wrong
                                    / correct
  sci_decisions_{tag}_text.jsonl      text System-One rows for --remote eval

Leakage control: split by pid — no passage appears in two splits.

Distractor/negative strategies:
  - same-passage swap: answer to a different question on the same pid
    (topically right context, factually wrong for this question)
  - number perturb: for answers containing numerals, multiply by {0.1,2,10}
    or perturb digits — hard negatives for the `numerical` type
  - cross-passage: answer to a question on a different pid (easy negative)

Record format (model path): {"ctx":[ids], "opts":[[ids]...], "gold":i, "qid":str}

v2 sources: --qa-raw GLOB accepts qa_raw_shard*.jsonl
{"pid","passage","qa":[{q,a}]} (ecoreasoner qa_gen output). Light
groundedness filters from qa_filter.py keep bad teacher generations out.
"""
import argparse
import glob
import json
import random
import re
from decimal import Decimal, localcontext
from itertools import chain
from pathlib import Path

from reverse_jev.data import (
    add_decision, content_fingerprint, exclude_record, group_records,
    input_fingerprints, normalized_content, partition_groups, read_jsonl,
    record_metadata, write_dataset,
)

INSTR_CHOICE = "Which answer correctly answers the question according to the supplied passage?"
INSTR_NOUL = "Does the proposed answer correctly answer the question according to the supplied passage?"
SCORE_LEGEND = ["unrelated answer", "related but wrong answer",
                "correct answer"]

NUM_RE = re.compile(r"(?<![\w.])[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?!\w)")
_WORD = re.compile(r"[a-záéíóúñü][a-záéíóúñü\-']*", re.I)
_STOP = set("""a an the of in on at to for and or but is are was were be been
it its this that these those with by from as we i you they he she not no do
does did have has had can could will would should may might than then so such
""".split())
MAX_CTX = 640          # leave headroom for opts inside seq_len=768
MAX_OPT = 120


def _filter_qa(row, policy, exclusions):
    if policy not in ("lexical", "schema"):
        raise ValueError("filter_policy must be lexical or schema")
    if any(not isinstance(row.get(key), str) or not row[key].strip()
           for key in ("passage", "q", "a")):
        exclude_record(exclusions, row, "invalid_qa_schema")
        return None
    row = dict(row, **{key: row[key].strip() for key in ("passage", "q", "a")},
               filter_policy=policy)
    if policy == "lexical" and not grounded(row["q"], row["a"], row["passage"]):
        exclude_record(exclusions, row, "lexical_consistency_filter")
        return None
    return row


def load_rows(path, filter_policy="lexical", exclusions=None, stats=None):
    recs = []
    for line_number, row in read_jsonl(path):
        if stats is not None:
            stats["input_records"] = stats.get("input_records", 0) + 1
            stats["qa_candidates"] = stats.get("qa_candidates", 0) + 1
        row = dict(row, source_file=str(path), source_line=line_number)
        row.setdefault("source", "ecoreasoner.qa_elite")
        kept = _filter_qa(row, filter_policy, exclusions)
        if kept is not None:
            recs.append(kept)
    return recs


def _content_words(text):
    return [w.lower() for w in _WORD.findall(text)
            if len(w) > 3 and w.lower() not in _STOP]


def _norm_num(s):
    text = s.replace("−", "-")
    if NUM_RE.fullmatch(text) is None:
        raise ValueError(f"unrecognized numeric literal: {s!r}")
    return Decimal(text.replace(",", ""))


def grounded(q, a, passage):
    """qa_filter-style gates: schema, answer recall >=0.5, number binding."""
    if any(not isinstance(text, str) or not text.strip() for text in (q, a, passage)):
        return False
    if len(q) > 400 or not (8 <= len(a) <= 512):
        return False
    pset = set(_content_words(passage))
    aw = _content_words(a)
    if aw and sum(1 for w in aw if w in pset) / len(aw) < 0.5:
        return False
    pnums = {_norm_num(n) for n in NUM_RE.findall(passage.replace("−", "-"))}
    return all(_norm_num(n) in pnums for n in NUM_RE.findall(a.replace("−", "-")))


def load_qa_raw(pattern, rng, qas_per_pid=2, min_recall=True, *,
                filter_policy=None, exclusions=None, stats=None):
    """qa_raw_shard*.jsonl -> flat recs {pid, passage, q, a}, filtered."""
    if type(qas_per_pid) is not int or qas_per_pid < 1:
        raise ValueError("qas_per_pid must be a positive integer")
    policy = filter_policy if filter_policy is not None else ("lexical" if min_recall else "schema")
    paths = sorted(glob.glob(str(pattern)))
    if not paths:
        raise ValueError(f"no input files match {pattern!r}")
    recs = []
    for fp in paths:
        for line_number, r in read_jsonl(fp):
            if stats is not None:
                stats["input_records"] = stats.get("input_records", 0) + 1
            r = dict(r, source_file=str(fp), source_line=line_number)
            r.setdefault("source", "ecoreasoner.qa_raw")
            if not isinstance(r.get("qa"), list):
                raise ValueError(f"{fp}:{line_number}: qa must be an array")
            qas = []
            for index, qa in enumerate(r["qa"]):
                if stats is not None:
                    stats["qa_candidates"] = stats.get("qa_candidates", 0) + 1
                if not isinstance(qa, dict):
                    raise ValueError(f"{fp}:{line_number}: qa[{index}] must be an object")
                row = {**{k: v for k, v in r.items() if k != "qa"}, **qa, "qa_index": index}
                kept = _filter_qa(row, policy, exclusions)
                if kept is not None:
                    qas.append(kept)
            rng.shuffle(qas)
            recs.extend(qas[:qas_per_pid])
            for row in qas[qas_per_pid:]:
                exclude_record(exclusions, row, "qas_per_pid_limit")
    return recs


def perturb_number(ans, rng):
    """Return a copy of `ans` with one number changed, or None."""
    ms = list(NUM_RE.finditer(ans.replace("−", "-")))
    if not ms:
        return None
    m = rng.choice(ms)
    x = _norm_num(m.group())
    with localcontext() as context:
        context.prec = max(28, len(x.as_tuple().digits) + 16)
        factor = Decimal(rng.choice(["0.1", "0.5", "2", "10"]))
        y = x * factor
        if y == x:          # degenerate (x==0 etc.)
            y = x + rng.choice([1, 2, 5])
        rep = f"{y:g}"
    return ans[:m.start()] + rep + ans[m.end():]


def prepare_records(recs, exclusions=None):
    rows = []
    for r in recs:
        if any(not isinstance(r.get(key), str) or not r[key].strip() for key in ("passage", "q", "a")):
            raise ValueError(f"invalid scientific QA record: {r.get('pid')!r}")
        rows.append(dict(r, **{key: r[key].strip() for key in ("passage", "q", "a")}))
    if not all("split_group" in r and "content_hash" in r for r in rows):
        rows = group_records(rows, "passage", source="ecoreasoner.qa")
    unique = {}
    for r in rows:
        r["sample_id"] = content_fingerprint([r["source"], r["passage"], r["q"], r["a"]])
        key = (r["content_hash"], normalized_content(r["q"]), normalized_content(r["a"]))
        if key in unique:
            exclude_record(exclusions, r, "duplicate_qa", retained_sample_id=unique[key]["sample_id"])
            unique[key].setdefault("duplicate_sources", []).append(record_metadata(r, r["source"], "unspecified"))
        else:
            unique[key] = r
    return list(unique.values())


def split_records(rows, rng, dev_frac=0.15, eval_frac=0.20, counts=None, exclusions=None):
    rows = prepare_records(rows, exclusions)
    return partition_groups(rows, rng, dev_frac, eval_frac, counts, exclusions)


def _candidate(answer, record, strategy, tag):
    return {"answer": answer, "provenance": {
        "strategy": strategy, "source_pid": record["pid"],
        "source_group": record["split_group"], "source": record["source"],
        "source_sample_id": record["sample_id"], "split": tag, "verified": False}}


def _cross_candidates(recs, record, positives, rng, tag):
    initial = rng.sample(range(len(recs)), min(16, len(recs)))
    visited, answers, candidates = set(initial), set(), []
    indices = chain(initial, (index for index in range(len(recs)) if index not in visited))
    for index in indices:
        other = recs[index]
        answer = normalized_content(other["a"])
        if other["split_group"] == record["split_group"] or answer in positives or answer in answers:
            continue
        candidates.append(_candidate(other["a"], other, "cross_passage", tag))
        answers.add(answer)
        if len(candidates) == 4:
            break
    return candidates


def build(recs, tok, rng, tag, *, exclusions=None, max_ctx=MAX_CTX, max_opt=MAX_OPT,
          overflow="error"):
    recs = prepare_records(recs, exclusions)
    if any(r.get("split", tag) != tag for r in recs):
        raise ValueError("negative pools must contain only records from the requested split")
    by_passage = {}
    for r in recs:
        by_passage.setdefault(r["content_hash"], []).append(r)
    id_rows, text_by_state = [], {}
    for r in recs:
        passage, q, gold_a = r["passage"], r["q"], r["a"]
        state = f"Passage: {passage}\nQuestion: {q}"
        metadata = dict(record_metadata(r, "ecoreasoner.qa", "unspecified"), split=tag,
                        evidence_span=[len("Passage: "), len("Passage: ") + len(passage)])
        identity = f"{tag}_{r['sample_id'][:20]}"
        positives = {normalized_content(o["a"]) for o in by_passage[r["content_hash"]]
                     if normalized_content(o["q"]) == normalized_content(q)}
        positive = _candidate(gold_a, r, "source_reference", tag)

        # ---------- choice: K=4 ----------
        sibs = [_candidate(o["a"], o, "same_passage_swap", tag)
                for o in by_passage[r["content_hash"]] if normalized_content(o["a"]) not in positives]
        sibs = list({normalized_content(c["answer"]): c for c in sibs}.values())
        rng.shuffle(sibs)
        distr = []
        distr += sibs[:2]                          # hard: same passage
        pert_answer = perturb_number(gold_a, rng)
        pert = (_candidate(pert_answer, r, "number_perturb", tag)
                if pert_answer and normalized_content(pert_answer) not in positives else None)
        if pert and all(normalized_content(c["answer"]) != normalized_content(pert_answer) for c in distr):
            distr.append(pert)                     # hard: perturbed gold
        cross = _cross_candidates(recs, r, positives, rng, tag)
        for c in cross:                           # fill: other passages
            if len(distr) >= 3:
                break
            if all(normalized_content(d["answer"]) != normalized_content(c["answer"]) for d in distr):
                distr.append(c)
        if len(distr) >= 3:
            opts = distr[:3] + [positive]
            rng.shuffle(opts)
            answers = [c["answer"] for c in opts]
            question = {"type": "choice", "instructions": INSTR_CHOICE,
                        "criteria": {answer: None for answer in answers}, "label": gold_a}
            meta = dict(metadata, label_status="heuristic", negative_provenance=[
                dict(c["provenance"], option_key=c["answer"]) for c in opts])
            add_decision(id_rows, text_by_state, tok, state, f"choice_{identity}",
                         question, answers.index(gold_a), meta, max_ctx, max_opt,
                         overflow, exclusions)
        else:
            exclude_record(exclusions, r, "insufficient_unique_distractors", split=tag, kind="choice",
                           available=len(distr), required=3)

        # ---------- noul + score: gold vs related-wrong vs unrelated ----
        related_wrong = pert or (sibs[0] if sibs else None)
        variants = [(positive, True, 2)]
        if related_wrong is not None:
            variants.append((related_wrong, False, 1))
        else:
            exclude_record(exclusions, r, "insufficient_related_negative", split=tag, kind="noul/score")
        unrelated = next((c for c in cross if related_wrong is None or
                          normalized_content(c["answer"]) != normalized_content(related_wrong["answer"])), None)
        if unrelated is not None:
            variants.append((unrelated, False, 0))
        else:
            exclude_record(exclusions, r, "insufficient_cross_passage_negative", split=tag, kind="noul/score")
        for j, (candidate, ok, sc) in enumerate(variants):
            st2 = f"{state}\nProposed answer: {candidate['answer']}"
            meta = dict(metadata, negative_provenance=candidate["provenance"],
                        label_status="source_reference" if ok else "heuristic")
            add_decision(id_rows, text_by_state, tok, st2, f"noul_{identity}_{j}",
                         {"type": "noul", "instructions": INSTR_NOUL, "label": ok},
                         0 if ok else 1, meta, max_ctx, max_opt, overflow, exclusions)
            add_decision(id_rows, text_by_state, tok, st2, f"score_{identity}_{j}",
                         {"type": "score", "instructions": "Rate the proposed answer.",
                          "criteria": SCORE_LEGEND, "label": sc}, sc,
                         dict(meta, label_status="heuristic_proxy", rubric_provenance="synthetic_strategy"),
                         max_ctx, max_opt, overflow, exclusions)
    return id_rows, list(text_by_state.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=None, help="qa_devin_elite.jsonl")
    ap.add_argument("--qa-raw", action="append", default=[],
                    help="glob of qa_raw_shard*.jsonl {pid,passage,qa}; "
                         "repeatable. Alternative to --qa")
    ap.add_argument("--qas-per-pid", type=int, default=2,
                    help="max QAs kept per passage (qa-raw mode)")
    ap.add_argument("--train-pids", type=int, default=None,
                    help="explicit train size in deduplicated source/content groups (else fractions)")
    ap.add_argument("--dev-pids", type=int, default=None)
    ap.add_argument("--eval-pids", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--eval-frac", type=float, default=0.20)
    ap.add_argument("--filter-policy", choices=("lexical", "schema"), default="lexical",
                    help="applies to elite and raw inputs; lexical consistency is not semantic verification")
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX)
    ap.add_argument("--max-opt", type=int, default=MAX_OPT)
    ap.add_argument("--overflow", choices=("error", "exclude", "truncate"),
                    default="exclude",
                    help="input over budget: fail, exclude with record, or "
                         "truncate with recorded counts")
    args = ap.parse_args()
    if bool(args.qa) == bool(args.qa_raw):
        ap.error("provide exactly one of --qa or --qa-raw")
    if args.train_pids is None and (args.dev_pids is not None or args.eval_pids is not None):
        ap.error("--train-pids is required when using explicit group counts")

    rng = random.Random(args.seed)
    out = Path(args.out)
    exclusions, stats, paths = [], {"input_records": 0, "qa_candidates": 0}, []
    if args.qa_raw:
        rows = []
        for pat in args.qa_raw:
            paths.extend(sorted(glob.glob(pat)))
            rows.extend(load_qa_raw(pat, rng, qas_per_pid=args.qas_per_pid,
                                   filter_policy=args.filter_policy, exclusions=exclusions, stats=stats))
    else:
        paths = [args.qa]
        rows = load_rows(args.qa, args.filter_policy, exclusions, stats)
    counts = None if args.train_pids is None else {
        "train": args.train_pids, "dev": args.dev_pids if args.dev_pids is not None else 0,
        "eval": args.eval_pids if args.eval_pids is not None else 0}
    splits = split_records(rows, random.Random(args.seed), args.dev_frac, args.eval_frac, counts, exclusions)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    files = {}
    for tag, recs in splits.items():
        id_rows, text_rows = build(recs, tok, random.Random(f"{args.seed}:{tag}"), tag,
                                   exclusions=exclusions, max_ctx=args.max_ctx,
                                   max_opt=args.max_opt, overflow=args.overflow)
        for kind in ("choice", "noul", "score"):
            files[out / f"sci_{kind}_{tag}.jsonl"] = [r for r in id_rows if r["kind"] == kind]
        files[out / f"sci_decisions_{tag}_text.jsonl"] = text_rows
    stats.update(filtered_records=len(rows), split_records={tag: len(rs) for tag, rs in splits.items()},
                 split_groups={tag: len({r["split_group"] for r in rs}) for tag, rs in splits.items()})
    manifest = {"builder": "build_sci_decisions", "seed": args.seed, "inputs": input_fingerprints(paths),
                "tokenizer": args.tokenizer, "max_ctx": args.max_ctx, "max_opt": args.max_opt,
                "overflow": args.overflow,
                "filter_policy": args.filter_policy, "filter_semantics": "lexical consistency, not semantic proof",
                "split_unit": "connected source identifiers and normalized passage content",
                "split_fractions": {"dev": args.dev_frac, "eval": args.eval_frac}, "requested_counts": counts,
                "negative_policy": "split-local heuristic alternatives; not human-verified negatives",
                "score_policy": "synthetic strategy proxy, not verified relevance or correctness",
                "counts": stats, "exclusions": exclusions}
    report = write_dataset(files, out / "sci_manifest.json", manifest)
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
