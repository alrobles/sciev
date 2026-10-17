#!/usr/bin/env python3
"""Anti-leakage training controls: reject gold when evidence is corrupted.

Complements build_evidence_controls.py (which keeps the *reference* gold
to measure agreement leakage) — this builder instead relabels destroyed
or swapped evidence with the *reject* label so heads learn to condition
on the passage:

  noul  -> index of "false"/not-supported (encoded order [true, false])
  score -> index 0 (lowest rubric level, "unrelated")

Choice is intentionally out of scope: with no evidence-removal option in
the criteria, a corrupted-evidence choice row has no valid reject gold —
the option-side leak there is an evaluation finding, not a trainable
supervision.

Usage:
    python data/build_antileak_train.py --text sci_decisions_train_text.jsonl \
        --kind noul --modes empty,shuffle \
        --out noul_antileak_controls.jsonl
    # then cat sci_noul_train.jsonl noul_antileak_controls.jsonl > mix.jsonl
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from sciev.data import (add_decision, exclude_record, input_fingerprints,
                        make_evidence_controls, read_jsonl, write_dataset)

MAX_CTX, MAX_OPT = 960, 120
REJECT_INDEX = {"noul": 1, "score": 0}  # [true,false] -> "no"; levels -> lowest
REJECT_SEMANTICS = {"noul": "reject_on_unsupported",
                    "score": "reject_lowest_rubric"}


def build_antileak(text_rows, tok, *, kind, modes, seed=7331, control_frac=1.0,
                   max_ctx=MAX_CTX, max_opt=MAX_OPT, overflow="exclude",
                   exclusions=None):
    exclusions = exclusions if exclusions is not None else []
    if kind not in REJECT_INDEX:
        raise ValueError(f"kind must be one of {sorted(REJECT_INDEX)}")
    if not 0 < control_frac <= 1.0:
        raise ValueError("control_frac must be in (0, 1]")
    pool = [r for r in text_rows
            if any(q.get("type") == kind for q in r.get("questions", {}).values())]
    if not pool:
        raise ValueError("no records carry a question of the requested kind")
    gold = REJECT_INDEX[kind]
    rng = random.Random(seed)
    id_rows, text_by_state, per_mode = [], {}, {}
    for mode in modes:
        controls = make_evidence_controls(pool, mode=mode, seed=seed)
        mode_rows = []
        for record in controls:
            meta = {key: val for key, val in record.items()
                    if key not in ("state", "questions")}
            meta["gold_semantics"] = REJECT_SEMANTICS[kind]
            meta["label_status"] = "antileak_train"
            for qid, question in record["questions"].items():
                if question.get("type") != kind:
                    continue
                noul_criteria = question.get("criteria")
                # noul rows may carry criteria=None (default yes/no encoding)
                if (kind == "noul" and noul_criteria is not None
                        and set(noul_criteria) != {"true", "false"}):
                    exclude_record(exclusions, meta, "noul_criteria_not_binary",
                                   qid=qid)
                    continue
                if kind == "score" and not 0 <= gold < len(question.get("criteria", ())):
                    exclude_record(exclusions, meta, "score_criteria_missing",
                                   qid=qid)
                    continue
                # the text-side write merges question over meta — keep the
                # anti-leak status consistent in both artifacts
                question["label_status"] = "antileak_train"
                question["gold_semantics"] = REJECT_SEMANTICS[kind]
                add_decision(mode_rows, text_by_state, tok, record["state"], qid,
                             question, gold, meta, max_ctx, max_opt, overflow,
                             exclusions)
        if control_frac < 1.0:
            keep = max(1, round(len(mode_rows) * control_frac))
            chosen = sorted(rng.sample(range(len(mode_rows)), keep))
            mode_rows = [mode_rows[i] for i in chosen]
        per_mode[mode] = len(mode_rows)
        id_rows.extend(mode_rows)
    return id_rows, list(text_by_state.values()), per_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="sci_decisions_*_text.jsonl")
    ap.add_argument("--kind", choices=sorted(REJECT_INDEX), required=True)
    ap.add_argument("--modes", default="empty,shuffle",
                    help="comma-separated evidence-corruption modes")
    ap.add_argument("--control-frac", type=float, default=1.0,
                    help="per-mode subsample fraction of control rows "
                         "(default 1.0 = keep all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX)
    ap.add_argument("--max-opt", type=int, default=MAX_OPT)
    ap.add_argument("--overflow", choices=("error", "exclude", "truncate"),
                    default="exclude")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes or any(m not in ("empty", "shuffle") for m in modes):
        ap.error("--modes must be a nonempty subset of empty,shuffle")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    text_rows = [row for _, row in read_jsonl(args.text)]
    exclusions = []
    id_rows, text_rows_out, per_mode = build_antileak(
        text_rows, tok, kind=args.kind, modes=modes, seed=args.seed,
        control_frac=args.control_frac,
        max_ctx=args.max_ctx, max_opt=args.max_opt, overflow=args.overflow,
        exclusions=exclusions)
    out = Path(args.out)
    manifest = {"builder": "build_antileak_train", "builder_version": 1,
                    "kind": args.kind, "modes": modes, "seed": args.seed,
                    "control_frac": args.control_frac,
                    "reject_index": REJECT_INDEX[args.kind],
                    "gold_semantics": REJECT_SEMANTICS[args.kind],
                    "inputs": input_fingerprints([args.text]),
                    "tokenizer": args.tokenizer,
                    "max_ctx": args.max_ctx, "max_opt": args.max_opt,
                    "overflow": args.overflow,
                    "exclusions": exclusions,
                    "counts": {"input_records": len(text_rows),
                               "rows": len(id_rows),
                               "per_mode": per_mode}}
    write_dataset({out: id_rows,
                   out.with_name(f"{out.stem}_text.jsonl"): text_rows_out},
                  out.with_suffix(".manifest.json"), manifest)
    print(f"[wrote] {out}: {len(id_rows)} rows {per_mode} "
          f"({len(exclusions)} excluded)", flush=True)


if __name__ == "__main__":
    main()
