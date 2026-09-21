"""convert_pairs.py — ecoreasoner pairs (ctx/ok/bad ids) -> System-One JSONL.

Each pair becomes a binary choice decision:

  {"state": <ctx text>, "questions": {"cont": {"type": "choice",
      "instructions": "Which continuation is inferentially correct?",
      "criteria": {"A": <ok text>, "B": <bad text>}, "label": "A"}}}

Option order (ok in A or B) is randomized per row, seed-fixed. Requires the
LLaDA tokenizer to detokenize the ids (HPC path or HF repo id).

  python data/convert_pairs.py --pairs /beegfs/.../pairs_L3.jsonl \
      --tokenizer GSAI-ML/LLaDA-8B-Instruct --out evals/pairs_L3_decisions.jsonl
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--instructions",
                    default="Which continuation is inferentially correct?")
    ap.add_argument("--qid", default="continuation")
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)

    n_in = n_out = 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for path in args.pairs:
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_in += 1
                ctx = tok.decode(rec["ctx"])
                ok, bad = tok.decode(rec["ok"]), tok.decode(rec["bad"])
                if not ctx.strip() or not ok.strip() or not bad.strip():
                    continue
                # randomize which slot holds the correct continuation
                if rng.random() < 0.5:
                    criteria, label = {"A": ok, "B": bad}, "A"
                else:
                    criteria, label = {"A": bad, "B": ok}, "B"
                fo.write(json.dumps({
                    "state": ctx,
                    "questions": {args.qid: {
                        "type": "choice",
                        "instructions": args.instructions,
                        "criteria": criteria,
                        "label": label}}},
                    ensure_ascii=False) + "\n")
                n_out += 1
    print(f"[convert] {n_in} pairs -> {n_out} decisions -> {args.out}")


if __name__ == "__main__":
    main()
