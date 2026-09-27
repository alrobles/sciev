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
import json
from pathlib import Path

from sciev.data import write_dataset
from sciev.decisions import encode_question, validate_decision_row

MAX_CTX, MAX_OPT = 640, 120
META_KEYS = ("split_group", "group_id", "pid", "source", "source_file",
             "source_line", "source_row", "group_scope", "content_hash",
             "sample_id", "reasoning_type", "meta", "label_status",
             "evidence_regime", "task", "split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--instructions", required=True,
                    help="| separated instruction variants")
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    variants = [s.strip() for s in args.instructions.split("|") if s.strip()]

    records = [json.loads(line) for line in open(args.text_file)]
    prefix = Path(args.out_prefix)
    for vi, instr in enumerate(variants, start=1):
        id_rows, text_rows = [], []
        skipped = 0
        for rec in records:
            questions = rec["questions"]
            assert len(questions) == 1, "variant builder expects one question per state"
            qid, q = next(iter(questions.items()))
            criteria = q["criteria"]
            option_keys = q.get("option_keys", list(criteria))
            gold = option_keys.index(q["label"])
            vq = {"type": "choice", "instructions": instr,
                  "criteria": criteria, "label": q["label"]}
            try:
                enc = encode_question(tok, rec["state"], vq,
                                      max_ctx=MAX_CTX, max_opt=MAX_OPT)
            except Exception:
                skipped += 1
                continue
            meta = {k: rec[k] for k in META_KEYS if k in rec}
            meta["instruction_variant"] = instr
            import hashlib
            from sciev.data import stable_json
            did = hashlib.sha256(stable_json(enc).encode()).hexdigest()
            row = validate_decision_row({**meta, **enc, "gold": gold,
                                         "qid": f"{qid}_v{vi}",
                                         "decision_id": did})
            id_rows.append(row)
            tq = {**{k: rec[k] for k in META_KEYS if k in rec},
                  "encoding": row["encoding"], "schema_version": row["schema_version"],
                  "instruction_variant": instr}
            text_rows.append({**tq, "state": rec["state"],
                              "questions": {f"{qid}_v{vi}": {**tq, **vq,
                                                            "kind": row["kind"],
                                                            "option_keys": row["option_keys"],
                                                            "decision_id": did}}})
        out = Path(f"{prefix}_instr{vi}.jsonl")
        write_dataset({out: id_rows, out.with_name(f"{out.stem}_text.jsonl"): text_rows},
                      out.with_suffix(".manifest.json"),
                      {"builder": "build_instr_variants",
                       "instruction_variant": instr,
                       "source_text_file": args.text_file,
                       "counts": {"rows": len(id_rows),
                                  "overflow_excluded": skipped}})
        print(f"[v{vi}] {instr!r} -> {out} ({len(id_rows)} rows, {skipped} overflow)")


if __name__ == "__main__":
    main()
