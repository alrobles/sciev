#!/usr/bin/env python3
"""Convert public benchmarks into sciev decision rows.

Outputs the same {ctx:[ids], opts:[[ids]...], gold, kind, qid} format as
build_sci_decisions.py, plus a text row file for --remote eval.

Benchmarks:
  gpqa    csv (Question, Correct Answer, Incorrect Answer 1-3) -> choice K=4
          ctx = "Question: ..." (no passage: parametric reasoning)
  scifact parquet (claim, title, abstract[list], verdict) -> noul K=2,
          score K=3. ctx = title + abstract + claim.
          noul: SUPPORT->yes, CONTRADICT/NEI->no
          score: SUPPORT->2, CONTRADICT->1, NEI->0

Usage:
  python convert_benchmarks.py --bench gpqa \
      --src bench_external/gpqa/dataset/gpqa_main.csv \
      --out bench_external/gpqa/gpqa_main_choice_eval.jsonl
  python convert_benchmarks.py --bench scifact \
      --src bench_external/scifact/validation.parquet \
      --out-prefix bench_external/scifact/scifact_dev
"""
import argparse
import csv
import json
import random
from pathlib import Path

from sciev.data import (
    add_decision, content_fingerprint, exclude_record, group_records,
    input_fingerprints, normalized_content, read_jsonl, record_metadata,
    write_dataset,
)

INSTR_CHOICE = "Which answer is correct?"
INSTR_NOUL = "Is the claim supported by the passage?"
SCORE_LEGEND = ["unrelated answer", "related but wrong answer",
                "correct answer"]
MAX_CTX = 640
MAX_OPT = 120


SCIFACT_CRITERIA = {
    "SUPPORT": "The passage supports the claim.",
    "CONTRADICT": "The passage contradicts the claim.",
    "NEI": "The passage provides insufficient evidence to decide the claim.",
}


def write_rows(path, id_rows, text_rows, manifest=None):
    path = Path(path)
    tp = path.with_name(f"{path.stem}_text.jsonl")
    report = write_dataset({path: id_rows, tp: text_rows}, path.with_suffix(".manifest.json"), manifest or {})
    print(f"[wrote] {report['outputs'][0]['path']}: {len(id_rows)}  (+text {tp.name})")
    return report


def _manifest(src, tok, rng, count, exclusions, max_ctx, max_opt, **extra):
    return {"builder": "convert_benchmarks", "inputs": input_fingerprints([src]),
            "tokenizer": getattr(tok, "name_or_path", type(tok).__name__),
            "rng_state_sha256": content_fingerprint(rng.getstate()),
            "max_ctx": max_ctx, "max_opt": max_opt, "counts": {"input_records": count},
            "exclusions": exclusions, **extra}


def _deduplicate(rows, identity, exclusions):
    unique = {}
    for row in rows:
        row["sample_id"] = content_fingerprint(identity(row))
        if row["sample_id"] in unique:
            exclude_record(exclusions, row, "duplicate_record", retained_sample_id=row["sample_id"])
            unique[row["sample_id"]].setdefault("duplicate_sources", []).append(
                record_metadata(row, row["source"], "unspecified"))
        else:
            unique[row["sample_id"]] = row
    return list(unique.values())


def _text_for_kind(text_rows, kind):
    selected = []
    for row in text_rows:
        questions = {qid: q for qid, q in row["questions"].items() if q["type"] == kind}
        if questions:
            first = next(iter(questions.values()))
            selected.append(dict(row, **{k: first[k] for k in ("task", "label_status", "legacy_proxy") if k in first},
                                 questions=questions))
    return selected


def conv_gpqa(src, out, tok, rng, *, max_ctx=MAX_CTX, max_opt=MAX_OPT, seed=None,
              overflow="exclude"):
    id_rows, text_by_state, exclusions = [], {}, []
    with open(src, newline="", encoding="utf-8") as f:
        inputs = list(csv.DictReader(f))
    manifest = _manifest(src, tok, rng, len(inputs), exclusions, max_ctx, max_opt, benchmark="gpqa", seed=seed,
                         overflow=overflow, evidence_regime="no_passage_parametric")
    rows = []
    fields = ("Question", "Correct Answer", *(f"Incorrect Answer {j}" for j in (1, 2, 3)))
    for i, r in enumerate(inputs):
        r = dict(r, source_file=str(src), source_row=i + 1)
        r.setdefault("source", "gpqa")
        if any(not isinstance(r.get(key), str) or not r[key].strip() for key in fields):
            exclude_record(exclusions, r, "invalid_gpqa_schema")
            continue
        r.update({key: r[key].strip() for key in fields})
        if len({normalized_content(r[key]) for key in fields[1:]}) != 4:
            exclude_record(exclusions, r, "duplicate_options")
            continue
        rows.append(r)
    rows = group_records(rows, "Question", scope="question", source="gpqa")
    rows = _deduplicate(rows, lambda r: [r["Question"], r["Correct Answer"],
                                       sorted(r[key] for key in fields[1:])], exclusions)
    for r in rows:
        q, gold = r["Question"], r["Correct Answer"]
        opts = [r[f"Incorrect Answer {j}"] for j in (1, 2, 3)] + [gold]
        rng.shuffle(opts)
        metadata = dict(record_metadata(r, "gpqa", "parametric_scientific_qa"),
                        meta=r.get("Subdomain", ""), label_status="source_reference",
                        evidence_regime="no_passage_parametric", task="gpqa_choice")
        add_decision(id_rows, text_by_state, tok, f"Question: {q}", f"gpqa_{r['sample_id'][:20]}",
                     {"type": "choice", "instructions": INSTR_CHOICE,
                      "criteria": {option: None for option in opts}, "label": gold},
                     opts.index(gold), metadata, max_ctx, max_opt, overflow, exclusions)
    manifest["counts"]["accepted_records"] = len(rows)
    return write_rows(out, id_rows, list(text_by_state.values()), manifest)


def conv_scifact(src, out_prefix, tok, rng, *, legacy_score=False,
                  max_ctx=MAX_CTX, max_opt=MAX_OPT, seed=None, overflow="exclude"):
    import pandas as pd
    inputs = pd.read_parquet(src).to_dict("records")
    id_rows, text_by_state, exclusions, rows = [], {}, [], []
    split = Path(src).stem
    manifest = _manifest(src, tok, rng, len(inputs), exclusions, max_ctx, max_opt, benchmark="scifact", seed=seed,
                         overflow=overflow,
                         primary_tasks=["support_vs_not", "nominal_verdict"], legacy_score=legacy_score,
                         split_unit="provided source split; document identifiers or normalized evidence passage")
    for i, r in enumerate(inputs):
        r = dict(r, source_file=str(src), source_row=i + 1)
        r.setdefault("source", "scifact")
        verdict = str(r.get("verdict", "")).strip().upper()           # SUPPORT / CONTRADICT / NEI
        if verdict not in SCIFACT_CRITERIA:
            raise ValueError(f"{src}:row {i + 1}: unknown SciFact verdict {verdict!r}")
        claim, title, sentences = r.get("claim"), r.get("title", ""), r.get("abstract")
        if hasattr(sentences, "tolist"):
            sentences = sentences.tolist()
        if (not isinstance(claim, str) or not claim.strip() or not isinstance(title, str)
                or not isinstance(sentences, (list, tuple)) or not sentences
                or any(not isinstance(s, str) for s in sentences) or not " ".join(sentences).strip()):
            exclude_record(exclusions, r, "invalid_scifact_schema")
            continue
        abstract = " ".join(sentences).strip()
        rows.append(dict(r, claim=claim.strip(), title=title.strip(), abstract=abstract, verdict=verdict,
                         evidence=f"{title.strip()}\n{abstract}"))
    rows = group_records(rows, "evidence", source="scifact")
    rows = _deduplicate(rows, lambda r: [r["evidence"], r["claim"], r["verdict"]], exclusions)
    for r in rows:
        claim, title, abstract, verdict = r["claim"], r["title"], r["abstract"], r["verdict"]
        evidence = f"Title: {title}\nAbstract: {abstract}"
        state = f"{evidence}\nClaim: {claim}"
        identity = f"sf_{split}_{r['sample_id'][:20]}"
        metadata = dict(record_metadata(r, "scifact", "evidence_verdict"), source_verdict=verdict,
                        label_status="source_reference", split=split, evidence_regime="provided_passage",
                        evidence_span=[0, len(evidence)])
        supported = verdict == "SUPPORT"
        add_decision(id_rows, text_by_state, tok, state, f"{identity}_n",
                     {"type": "noul", "instructions": INSTR_NOUL, "label": supported,
                      "criteria": {"true": "The passage supports the claim.",
                                   "false": "The passage contradicts the claim or provides insufficient evidence."}},
                     0 if supported else 1, dict(metadata, task="scifact_support_vs_not"),
                     max_ctx, max_opt, overflow, exclusions)
        add_decision(id_rows, text_by_state, tok, state, f"{identity}_c",
                     {"type": "choice", "instructions": "How does the passage relate to the claim?",
                      "criteria": SCIFACT_CRITERIA, "label": verdict}, list(SCIFACT_CRITERIA).index(verdict),
                     dict(metadata, task="scifact_nominal_verdict", label_space=list(SCIFACT_CRITERIA)),
                     max_ctx, max_opt, overflow, exclusions)
        if legacy_score:
            score_gold = {"SUPPORT": 2, "CONTRADICT": 1, "NEI": 0}[verdict]
            add_decision(id_rows, text_by_state, tok, state, f"{identity}_s",
                         {"type": "score", "instructions": "Rate the claim.",
                          "criteria": SCORE_LEGEND, "label": score_gold}, score_gold,
                         dict(metadata, task="scifact_legacy_ordinal_proxy", legacy_proxy=True,
                              label_status="legacy_proxy", rubric_provenance="legacy_verdict_mapping"),
                         max_ctx, max_opt, overflow, exclusions)
    files = {}
    for kind in ("noul", "choice", *(("score",) if legacy_score else ())):
        task_name = "score_legacy_proxy" if kind == "score" else kind
        path = Path(f"{out_prefix}_{task_name}_eval.jsonl")
        files[path] = [row for row in id_rows if row["kind"] == kind]
        files[path.with_name(f"{path.stem}_text.jsonl")] = _text_for_kind(text_by_state.values(), kind)
    manifest["counts"]["accepted_records"] = len(rows)
    return write_dataset(files, Path(f"{out_prefix}_manifest.json"), manifest)


def conv_classification(src, out, tok, rng, cfg, *, max_ctx=MAX_CTX, max_opt=MAX_OPT, seed=None,
                        overflow="exclude"):
    """Generic K-way classification -> choice rows.

    cfg: {text_field, label_names, ctx_fmt, question, label_field?,
          labels_file?}. Options = label_names in fixed order, gold = label.
    """
    labels = cfg.get("label_names")
    if cfg.get("labels_file"):
        labels = [x.replace("_", " ")
                  for x in json.loads(Path(cfg["labels_file"]).read_text())]
    if (not isinstance(labels, list) or len(labels) < 2
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("label_names must contain at least two distinct nonempty strings")
    if Path(src).suffix == ".parquet":
        import pandas as pd
        inputs = pd.read_parquet(src).to_dict("records")
    else:
        inputs = [dict(row, source_line=line) for line, row in read_jsonl(src)]
    id_rows, text_by_state, exclusions, rows = [], {}, [], []
    tag, label_field = cfg.get("tag", Path(src).stem), cfg.get("label_field", "label")
    manifest = _manifest(src, tok, rng, len(inputs), exclusions, max_ctx, max_opt, benchmark=tag, seed=seed,
                         overflow=overflow, label_space=labels)
    if cfg.get("labels_file"):
        manifest["inputs"] = input_fingerprints([src, cfg["labels_file"]])
    for i, r in enumerate(inputs):
        r = dict(r, source_file=str(src), source_row=i + 1)
        r.setdefault("source", tag)
        text, gold = r.get(cfg["text_field"]), r.get(label_field)
        if type(gold) is not int or not 0 <= gold < len(labels):
            raise ValueError(f"{src}:row {i + 1}: invalid classification label {gold!r}")
        if not isinstance(text, str) or not text.strip():
            exclude_record(exclusions, r, "invalid_classification_text")
            continue
        rows.append(dict(r, **{cfg["text_field"]: text.strip()}))
    rows = group_records(rows, cfg["text_field"], scope="text", source=tag)
    rows = _deduplicate(rows, lambda r: [tag, r[cfg["text_field"]], r[label_field]], exclusions)
    for r in rows:
        text, gold = r[cfg["text_field"]], r[label_field]
        state = cfg["ctx_fmt"].format(t=text)
        metadata = dict(record_metadata(r, tag, "classification"), label_status="source_reference", label_space=labels)
        add_decision(id_rows, text_by_state, tok, state, f"{tag}_{r['sample_id'][:20]}",
                     {"type": "choice", "instructions": cfg["question"],
                      "criteria": {label: None for label in labels}, "label": labels[gold]},
                     gold, metadata, max_ctx, max_opt, overflow, exclusions)
    manifest["counts"]["accepted_records"] = len(rows)
    return write_rows(out, id_rows, list(text_by_state.values()), manifest)


CLASSIF_CFGS = {
    "sst2": {"text_field": "sentence", "label_field": "label",
             "label_names": ["negative", "positive"],
             "ctx_fmt": "Text: {t}",
             "question": "What is the sentiment of this text?"},
    "ag_news": {"text_field": "text", "label_field": "label",
                "label_names": ["World", "Sports", "Business",
                                "Science/Technology"],
                "ctx_fmt": "Article: {t}",
                "question": "Which category does this article belong to?"},
    "banking77": {"text_field": "text", "label_field": "label",
                  "ctx_fmt": "Customer message: {t}",
                  "question": "What is the customer's intent?"},
    "enron": {"text_field": "text", "label_field": "label",
              "label_names": ["not spam", "spam"],
              "ctx_fmt": "Email: {t}",
              "question": "Is this email spam?"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["gpqa", "scifact",
                                        *CLASSIF_CFGS], required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=None, help="gpqa: output jsonl")
    ap.add_argument("--out-prefix", default=None,
                    help="scifact: prefix for native _noul/_choice_eval.jsonl")
    ap.add_argument("--labels-file", default=None,
                    help="json list of label names (banking77)")
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--legacy-score", action="store_true",
                    help="also emit the explicitly legacy/proxy SciFact ordinal mapping")
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX)
    ap.add_argument("--max-opt", type=int, default=MAX_OPT)
    ap.add_argument("--overflow", choices=("error", "exclude", "truncate"),
                    default="exclude",
                    help="input over budget: fail, exclude with record, or "
                         "truncate with recorded counts")
    args = ap.parse_args()
    if args.bench == "scifact" and not args.out_prefix:
        ap.error("--out-prefix is required for scifact")
    if args.bench != "scifact" and (not args.out or args.legacy_score):
        ap.error("--out is required; --legacy-score is only available for scifact")

    rng = random.Random(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    options = {"max_ctx": args.max_ctx, "max_opt": args.max_opt, "seed": args.seed,
               "overflow": args.overflow}
    if args.bench == "gpqa":
        conv_gpqa(args.src, args.out, tok, rng, **options)
    elif args.bench == "scifact":
        conv_scifact(args.src, args.out_prefix, tok, rng, legacy_score=args.legacy_score, **options)
    else:
        cfg = dict(CLASSIF_CFGS[args.bench], tag=args.bench)
        if args.labels_file:
            cfg["labels_file"] = args.labels_file
        conv_classification(args.src, args.out, tok, rng, cfg, **options)


if __name__ == "__main__":
    main()
