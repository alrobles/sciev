"""Aggregate instruction-variant eval JSONs into a prompt ensemble.

Matches per-item predictions across variant evals by split_group (question
content hash, invariant to instruction wording), averages probability
vectors, and reports per-variant + ensemble accuracy/ECE/agreement.

Usage:
    python scripts/ensemble_instr_variants.py \
        --evals eval_r2_gpqa_main_clean.json eval_c_choice_gpqa_main_instr{1..4}.json \
        --out ensemble_gpqa_main.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def load_preds(path):
    r = json.load(open(path))
    de = r.get("decisions_eval", r)
    preds = de.get("predictions")
    assert preds, f"no predictions in {path}"
    return {p["split_group"]: p for p in preds}, de


def ece(items, nbins=10):
    """items: list of (confidence, correct)."""
    if not items:
        return None
    tot, bins = len(items), defaultdict(list)
    for conf, ok in items:
        bins[min(nbins - 1, int(conf * nbins))].append((conf, ok))
    return sum(len(b) / tot * abs(sum(c for c, _ in b) / len(b)
                                   - sum(o for _, o in b) / len(b))
               for b in bins.values() if b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    per_file = []
    by_group = {}
    for path in args.evals:
        preds, de = load_preds(path)
        groups = set(preds)
        per_file.append({"file": Path(path).name, "n": de.get("n"),
                         "acc": de.get("acc"), "ece": de.get("ece"),
                         "temp": de.get("temperature"),
                         "elapsed_s": de.get("elapsed_s")})
        for g, p in preds.items():
            by_group.setdefault(g, []).append(p)

    common = {g for g, ps in by_group.items()
              if len(ps) == len(args.evals)}
    print(f"variant files: {len(args.evals)}; items with full coverage: {len(common)}")

    ens_correct, ens_items, agree_all, variant_items = 0, [], 0, defaultdict(list)
    flips = Counter()
    for g in sorted(common):
        ps = by_group[g]
        keys = ps[0]["option_keys"]
        assert all(p["option_keys"] == keys for p in ps)
        assert all(p["gold"] == ps[0]["gold"] for p in ps)
        probs = [sum(p["probabilities"][i] for p in ps) / len(ps)
                 for i in range(len(keys))]
        pred = max(range(len(keys)), key=lambda i: probs[i])
        ok = pred == ps[0]["gold"]
        ens_correct += ok
        ens_items.append((probs[pred], ok))
        preds_each = {p["prediction"] for p in ps}
        agree_all += len(preds_each) == 1
        for pi, p in enumerate(ps):
            variant_items[pi].append((p["max_probability"], p["correct"]))
        if len(preds_each) > 1:
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    flips[(ps[i]["prediction"], ps[j]["prediction"])] += 1

    n = len(common)
    report = {
        "n_variants": len(args.evals),
        "n_common_items": n,
        "per_variant": per_file,
        "per_variant_acc_on_common": {
            str(i): round(sum(o for _, o in items) / len(items), 4)
            for i, items in sorted(variant_items.items())},
        "ensemble": {"acc": round(ens_correct / n, 4),
                     "ece": round(ece(ens_items), 4),
                     "mean_conf": round(sum(c for c, _ in ens_items) / n, 4),
                     "full_agreement_rate": round(agree_all / n, 4)},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
