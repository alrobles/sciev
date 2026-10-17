"""Build instruction-variant decision files from a clean _text.jsonl eval set.

Each variant re-encodes every text record with a semantically equivalent
choice instruction, preserving options/gold/sample_id so predictions can be
ensembled per item across variants. For prompt-ensemble diagnostics.

Usage:
    python data/build_instr_variants.py \
        --text-file .../gpqa_main_choice_eval_text.jsonl \
        --out-prefix .../gpqa_main_choice_eval \
        --instructions "Select the best answer.|Choose the correct option.|Identify the correct answer."
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sciev.data import stable_json, write_dataset
from sciev.decisions import InputOverflow, encode_question, validate_decision_row
from sciev.protocol import file_fingerprint

MAX_CTX, MAX_OPT = 640, 120
META_KEYS = ("split_group", "group_id", "pid", "source", "source_file",
             "source_line", "source_row", "group_scope", "content_hash",
             "sample_id", "reasoning_type", "meta", "label_status",
             "evidence_regime", "task", "split")


def build_variant(records, tok, instruction, index, max_ctx=MAX_CTX, max_opt=MAX_OPT,
                  exclude_overflow=False):
    ids, texts, exclusions = [], [], []
    for rec in records:
        if len(rec["questions"]) != 1:
            raise ValueError("variant builder requires one question per state")
        qid, question = next(iter(rec["questions"].items()))
        if question.get("type") != "choice":
            raise ValueError("instruction variants require choice questions")
        variant = {"type": "choice", "instructions": instruction,
                   "criteria": question["criteria"], "label": question["label"]}
        try:
            encoded = encode_question(tok, rec["state"], variant, max_ctx=max_ctx, max_opt=max_opt)
        except InputOverflow:
            if not exclude_overflow:
                raise
            exclusions.append({"qid": qid, "sample_id": rec.get("sample_id"), "reason": "input_overflow"})
            continue
        metadata = {key: question.get(key, rec.get(key)) for key in META_KEYS
                    if key in question or key in rec}
        metadata["instruction_variant"] = instruction
        identity = hashlib.sha256(stable_json(encoded).encode()).hexdigest()
        qid = f"{qid}_v{index}"
        row = validate_decision_row({**metadata, **encoded, "gold": encoded["option_keys"].index(question["label"]),
                                     "qid": qid, "decision_id": identity})
        ids.append(row)
        text_meta = {**metadata, "encoding": row["encoding"], "schema_version": row["schema_version"]}
        texts.append({**text_meta, "state": rec["state"], "questions": {
            qid: {**text_meta, **variant, "kind": row["kind"], "option_keys": row["option_keys"], "decision_id": identity}}})
    return ids, texts, exclusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--instructions", required=True, help="| separated instruction variants")
    ap.add_argument("--include-original", action="store_true")
    ap.add_argument("--exclude-overflow", action="store_true")
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX)
    ap.add_argument("--max-opt", type=int, default=MAX_OPT)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    with Path(args.text_file).open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    variants = [text.strip() for text in args.instructions.split("|") if text.strip()]
    if not records or not variants:
        ap.error("records and instruction variants must be nonempty")
    if args.include_original:
        original = {question["instructions"] for rec in records for question in rec["questions"].values()}
        if len(original) != 1:
            ap.error("original instructions differ across the source cohort")
        variants.insert(0, original.pop())
    outputs = []
    for index, instruction in enumerate(variants, start=0 if args.include_original else 1):
        ids, texts, exclusions = build_variant(records, tok, instruction, index,
                                               args.max_ctx, args.max_opt, args.exclude_overflow)
        path = Path(f"{args.out_prefix}_instr{index}.jsonl")
        manifest = {"builder": "build_instr_variants", "builder_version": 2,
                    "instruction_variant": instruction, "source_text_file": file_fingerprint(args.text_file),
                    "tokenizer": args.tokenizer, "max_ctx": args.max_ctx, "max_opt": args.max_opt,
                    "exclusions": exclusions, "counts": {"input_records": len(records), "rows": len(ids)}}
        outputs.append((path, ids, texts, manifest))
    for path, ids, texts, manifest in outputs:
        write_dataset({path: ids, path.with_name(f"{path.stem}_text.jsonl"): texts},
                      path.with_suffix(".manifest.json"), manifest)
        print(f"[wrote] {path}: {len(ids)} rows", flush=True)


if __name__ == "__main__":
    main()
