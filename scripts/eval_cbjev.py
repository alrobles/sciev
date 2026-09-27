"""Evaluate cbjev (encoder decision layer) on sciev text-format decision files.

Consumes *_text.jsonl records ({state, questions}) produced by
data/convert_benchmarks.py and emits a report JSON with the same metric
fields as sciev.eval (acc, ece, nll, brier, per-item predictions) so
round2_analysis.py can compare directly.

Usage:
    python scripts/eval_cbjev.py --file <text.jsonl> --kind choice|noul|score \
        --out <report.json> [--order-votes 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def build_call(row):
    """Map a sciev text record to (state, questions, qid, gold_key, kind)."""
    state = row["state"]
    qid, q = next(iter(row["questions"].items()))
    kind = q["type"]
    spec = {"type": kind, "instructions": q["instructions"]}
    if kind == "choice":
        # criteria: {option_key: description-or-null} — keep the labels
        spec["criteria"] = dict(q["criteria"])
        gold_key = q["label"]
    elif kind == "noul":
        gold_key = "true" if q["label"] in (True, "true", 1) else "false"
    elif kind == "score":
        spec["criteria"] = list(q["criteria"])
        gold_key = str(q["label"])
    else:
        raise ValueError(kind)
    return state, {qid: spec}, qid, gold_key, kind


def metrics(records):
    n = len(records)
    acc = float(np.mean([r["correct"] for r in records])) if n else 0.0
    nll = brier = 0.0
    for r in records:
        p = max(r["probabilities"].get(r["gold_key"], 0.0), 1e-12)
        nll -= np.log(p)
        brier += (r["max_probability"] - float(r["correct"])) ** 2
    # adaptive ECE over 10 equal-count bins
    conf = np.array([r["max_probability"] for r in records])
    cor = np.array([r["correct"] for r in records])
    ece = 0.0
    if n:
        order = np.argsort(conf)
        for b in np.array_split(order, 10):
            if len(b):
                ece += (len(b) / n) * abs(conf[b].mean() - cor[b].mean())
    return {"n": n, "acc": round(acc, 4), "ece": round(float(ece), 4),
            "nll": round(nll / n, 4) if n else None,
            "brier": round(brier / n, 4) if n else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--kind", required=True,
                    choices=["choice", "noul", "score"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--order-votes", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import cbjev

    rows = [json.loads(l) for l in open(args.file)]
    if args.limit:
        rows = rows[: args.limit]

    agent = cbjev.load(order_votes=args.order_votes)

    records, t0 = [], time.time()
    for i, row in enumerate(rows):
        state, questions, qid, gold_key, kind = build_call(row)
        if kind != args.kind:
            continue
        try:
            res = agent.predict(state, questions)["answers"][qid]
        except Exception as exc:  # noqa: BLE001
            records.append({"qid": qid, "gold_key": gold_key, "error": repr(exc),
                            "correct": 0, "max_probability": 0.0,
                            "probabilities": {}})
            continue
        if kind == "choice":
            probs = res["probabilities"]
            pred = max(probs, key=probs.get)
        elif kind == "noul":
            probs = {"true": res["noul"], "false": 1.0 - res["noul"]}
            pred = "true" if res["noul"] >= 0.5 else "false"
        else:
            probs = res["probabilities"]
            pred = max(probs, key=probs.get)
        records.append({"qid": qid, "gold_key": gold_key,
                        "prediction": pred,
                        "probabilities": probs,
                        "max_probability": float(max(probs.values())),
                        "correct": int(pred == gold_key)})
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(rows)}] {time.time()-t0:.0f}s", file=sys.stderr)

    out = {"engine": "cbjev", "file": args.file, "kind": args.kind,
           "order_votes": args.order_votes, "elapsed_s": round(time.time() - t0, 1),
           **metrics(records), "predictions": records}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "predictions"},
                     indent=1))


if __name__ == "__main__":
    main()
