#!/usr/bin/env python3
"""Build evidence-control decision sets from text System-One rows.

Input: sci_decisions_{split}_text.jsonl produced by build_sci_decisions.py
(text requests: state + questions, carrying evidence_span / split_group /
split provenance).

Controls (reverse_jev.data.make_evidence_controls):
  empty   — the evidence span is removed
  shuffle — the evidence span is replaced by a donor passage from a
            *different* source group within the same split

Controls never receive a new ground truth. The encoded rows score
*agreement with the reference label*: if a model still selects the
originally-correct answer after the evidence is destroyed or swapped, it
is not conditioning on the evidence. Every row is therefore marked
label_status=evidence_control and gold_semantics=reference_agreement.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from reverse_jev.data import (
    add_decision, exclude_record, input_fingerprints, make_evidence_controls,
    read_jsonl, write_dataset,
)

MAX_CTX = 960
MAX_OPT = 120
TRUTHY = (True, "true", "yes", "True", "Yes", 1)


def reference_gold_index(question):
    """Position of the reference answer inside the question's option order.

    choice/score: index of reference_label within criteria keys.
    noul: encoded order is [supported, not_supported] -> 0 if truthy else 1.
    """
    ref = question["reference_label"]
    kind = question["type"]
    if kind == "noul":
        return 0 if ref in TRUTHY else 1
    if kind == "score":
        gold = int(ref)
        if not 0 <= gold < len(question["criteria"]):
            raise ValueError("score reference out of rubric range")
        return gold
    return list(question["criteria"].keys()).index(ref)


def build(text_rows, tok, *, mode, seed, max_ctx, max_opt, overflow, exclusions):
    controls = make_evidence_controls(text_rows, mode=mode, seed=seed)
    id_rows, text_by_state, skipped = [], {}, 0
    for record in controls:
        meta = {key: val for key, val in record.items() if key not in ("state", "questions")}
        meta["gold_semantics"] = "reference_agreement"
        for qid, question in record["questions"].items():
            try:
                gold = reference_gold_index(question)
            except (ValueError, KeyError, TypeError):
                exclude_record(exclusions, meta, "reference_label_not_in_options",
                               qid=qid, kind=question.get("type"))
                skipped += 1
                continue
            add_decision(id_rows, text_by_state, tok, record["state"], qid, question,
                         gold, meta, max_ctx, max_opt, overflow, exclusions)
    return id_rows, list(text_by_state.values()), len(controls), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="sci_decisions_*_text.jsonl source rows")
    ap.add_argument("--out", required=True,
                    help="output directory; writes sci_controls_{kind}_{mode}_eval.jsonl")
    ap.add_argument("--mode", choices=("empty", "shuffle"), required=True)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX)
    ap.add_argument("--max-opt", type=int, default=MAX_OPT)
    ap.add_argument("--overflow", choices=("error", "exclude", "truncate"), default="exclude")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    text_rows = [rec for _, rec in read_jsonl(args.text)]
    exclusions = []
    id_rows, controlled_text, n_controls, skipped = build(
        text_rows, tok, mode=args.mode, seed=args.seed,
        max_ctx=args.max_ctx, max_opt=args.max_opt, overflow=args.overflow,
        exclusions=exclusions)

    out = Path(args.out)
    files = {}
    by_kind = {}
    for row in id_rows:
        by_kind.setdefault(row["kind"], []).append(row)
    for kind, rows in sorted(by_kind.items()):
        files[str(out / f"sci_controls_{kind}_{args.mode}_eval.jsonl")] = rows
    files[str(out / f"sci_controls_{args.mode}_text.jsonl")] = controlled_text

    manifest = {
        "schema": "evidence_controls-v1",
        "mode": args.mode, "seed": args.seed,
        "source_text": args.text,
        "inputs": input_fingerprints([args.text]),
        "tokenizer": args.tokenizer,
        "max_ctx": args.max_ctx, "max_opt": args.max_opt, "overflow": args.overflow,
        "counts": {"input_text_records": len(text_rows),
                   "controlled_records": n_controls,
                   "skipped_reference_label": skipped},
        "exclusions": exclusions,
        "gold_semantics": "reference_agreement — row.gold is the position of the "
                          "original reference answer, NOT a new ground truth for "
                          "the altered evidence",
    }
    report = write_dataset(files, out / f"sci_controls_{args.mode}_manifest.json", manifest)
    print(json.dumps(report["counts"]["output_rows"], indent=2))
    print("exclusions:", json.dumps(report["counts"]["exclusions_by_reason"]))


if __name__ == "__main__":
    main()
