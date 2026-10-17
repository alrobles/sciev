"""Build grounded GPQA eval variants: closed-book state -> evidence-in-state.

Joins a clean *_text.jsonl eval file back to the GPQA source CSV via
source_row and appends an evidence block to the state text, re-encoding
only ctx — options/gold are byte-identical, so each grounded row pairs
with its closed-book counterpart by pid/sample_id:

  oracle : the expert-written `Explanation` — gold-revealing upper bound;
           measures whether the decision layer can exploit evidence at all
  nev    : first nonempty `Explanation_NEV_{1,2,3}` — a non-expert
           validator's own reasoning trace (noisy; written without gold)

Usage:
    python data/build_gpqa_grounded.py \
        --csv .../dataset/gpqa_main.csv \
        --text-file .../gpqa_main_choice_eval_clean_text.jsonl \
        --arm oracle --out .../gpqa_main_choice_eval_oracle.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sciev.data import input_fingerprints, stable_json, write_dataset
from sciev.decisions import InputOverflow, encode_question, validate_decision_row
from sciev.protocol import file_fingerprint

MAX_CTX, MAX_OPT = 896, 120  # wider than clean: evidence must survive
META_KEYS = ("split_group", "group_id", "pid", "source", "source_file",
             "source_line", "source_row", "group_scope", "content_hash",
             "sample_id", "reasoning_type", "meta", "label_status",
             "evidence_regime", "task", "split")
ARMS = ("oracle", "nev")


def _evidence(csv_row, arm):
    if arm == "oracle":
        return (csv_row.get("Explanation") or "").strip(), "Explanation"
    for index in (1, 2, 3):
        field = f"Explanation_NEV_{index}"
        text = (csv_row.get(field) or "").strip()
        if text:
            return text, field
    return "", None


def build_grounded(records, csv_rows, tok, arm, max_ctx=MAX_CTX, max_opt=MAX_OPT):
    ids, texts, exclusions = [], [], []
    for rec in records:
        qid, question = next(iter(rec["questions"].items()))
        if len(rec["questions"]) != 1 or question.get("type") != "choice":
            raise ValueError("grounded variants require one choice question per state")
        csv_row = csv_rows[int(rec["source_row"]) - 1]
        evidence, field = _evidence(csv_row, arm)
        if not evidence:
            exclusions.append({"qid": qid, "sample_id": rec.get("sample_id"),
                               "reason": f"no_{arm}_evidence"})
            continue
        state = f"{rec['state']}\n\nEvidence: {evidence}"
        try:
            encoded = encode_question(tok, state, question,
                                      max_ctx=max_ctx, max_opt=max_opt,
                                      overflow="truncate")
        except InputOverflow:
            raise AssertionError("overflow=truncate must not raise")
        metadata = {key: question.get(key, rec.get(key)) for key in META_KEYS
                    if key in question or key in rec}
        metadata.update(evidence_regime=f"{arm}_explanation",
                        task="gpqa_choice_grounded",
                        evidence_arm=arm, evidence_field=field)
        identity = hashlib.sha256(stable_json(encoded).encode()).hexdigest()
        grounded_qid = f"{qid}_gr-{arm}"
        row = validate_decision_row({**metadata, **encoded,
                                     "gold": encoded["option_keys"].index(question["label"]),
                                     "qid": grounded_qid, "decision_id": identity})
        ids.append(row)
        text_meta = {**metadata, "encoding": row["encoding"],
                     "schema_version": row["schema_version"]}
        texts.append({**text_meta, "state": state, "questions": {
            grounded_qid: {**text_meta, **question, "kind": row["kind"],
                           "option_keys": row["option_keys"],
                           "decision_id": identity}}})
    return ids, texts, exclusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="GPQA source csv")
    ap.add_argument("--text-file", required=True, help="clean *_text.jsonl")
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX)
    ap.add_argument("--max-opt", type=int, default=MAX_OPT)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    with Path(args.csv).open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    records = [json.loads(line) for line in Path(args.text_file).open()
               if line.strip()]
    if not records or not csv_rows:
        ap.error("csv and text-file must be nonempty")
    ids, texts, exclusions = build_grounded(records, csv_rows, tok, args.arm,
                                          args.max_ctx, args.max_opt)
    truncated = sum(1 for row in ids
                    if (row.get("truncation") or {}).get("context_tokens"))
    manifest = {"builder": "build_gpqa_grounded", "builder_version": 1,
                    "evidence_arm": args.arm,
                    "inputs": input_fingerprints([args.csv, args.text_file]),
                    "tokenizer": args.tokenizer,
                    "max_ctx": args.max_ctx, "max_opt": args.max_opt,
                    "exclusions": exclusions,
                    "counts": {"input_records": len(records), "rows": len(ids),
                               "ctx_truncated": truncated}}
    out = Path(args.out)
    write_dataset({out: ids,
                   out.with_name(f"{out.stem}_text.jsonl"): texts},
                  out.with_suffix(".manifest.json"), manifest)
    print(f"[wrote] {out}: {len(ids)} rows ({truncated} ctx-truncated, "
          f"{len(exclusions)} excluded)", flush=True)


if __name__ == "__main__":
    main()
