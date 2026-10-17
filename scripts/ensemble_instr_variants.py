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
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sciev.metrics import classification_metrics, evaluate_acceptance_policy, fit_acceptance_policy
from sciev.protocol import file_fingerprint


def load_preds(path):
    with Path(path).open() as stream:
        report = json.load(stream)
    data = report.get("decisions_eval", report)
    predictions = data.get("predictions")
    if not predictions:
        raise ValueError(f"no predictions in {path}")
    result = {}
    for row in predictions:
        keys = row.get("option_keys")
        group = row.get("split_group")
        if not group or not isinstance(keys, list) or len(set(keys)) != len(keys):
            raise ValueError("ensemble requires question identities and distinct option keys")
        classification_metrics([row["probabilities"]], [row["gold"]], predictions=[row["prediction"]])
        if len(keys) != len(row["probabilities"]):
            raise ValueError("option keys and probabilities must align")
        key = hashlib.sha256(json.dumps([group, sorted(keys)], ensure_ascii=False).encode()).hexdigest()
        if key in result:
            raise ValueError(f"duplicate item identity in {path}: {key}")
        result[key] = row
    return result, data


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


def paired_delta(reference, candidate, groups, seed=7331):
    deltas = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    grouped = defaultdict(list)
    for group, delta in zip(groups, deltas):
        grouped[group].append(float(delta))
    totals = np.asarray([sum(values) for values in grouped.values()])
    counts = np.asarray([len(values) for values in grouped.values()])
    draws = np.random.default_rng(seed).integers(len(grouped), size=(2000, len(grouped)))
    samples = totals[draws].sum(1) / counts[draws].sum(1)
    return {"delta_acc": float(deltas.mean()), "improved": int((deltas > 0).sum()),
            "regressed": int((deltas < 0).sum()), "bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "bootstrap_unit": "question_group", "bootstrap_samples": 2000, "seed": seed,
            "exploratory": True}


def aggregate(paths, method="probability_mean", allow_intersection=False):
    if len(paths) < 2 or len({str(Path(path).resolve()) for path in paths}) != len(paths):
        raise ValueError("an ensemble requires at least two distinct prediction files")
    if method not in ("probability_mean", "log_probability_mean"):
        raise ValueError("unsupported ensemble method")
    loaded = [load_preds(path) for path in paths]
    sets = [set(predictions) for predictions, _ in loaded]
    common = set.intersection(*sets)
    if not common:
        raise ValueError("ensemble has no common items")
    if not allow_intersection and any(keys != common for keys in sets):
        raise ValueError("variant cohorts differ; explicitly allow intersection to report exclusions")
    records, per_probs, per_golds, per_corrects = [], [[] for _ in paths], [], [[] for _ in paths]
    agreements, groups = 0, []
    for identity in sorted(common):
        rows = [predictions[identity] for predictions, _ in loaded]
        base = rows[0]
        keys = base["option_keys"]
        gold_key = keys[base["gold"]]
        if any(row["option_keys"][row["gold"]] != gold_key for row in rows):
            raise ValueError(f"conflicting gold identity for {identity}")
        aligned = np.asarray([[row["probabilities"][row["option_keys"].index(key)] for key in keys] for row in rows])
        if method == "probability_mean":
            probabilities = aligned.mean(0)
        else:
            logits = np.log(aligned.clip(min=1e-300)).mean(0)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
        prediction = min(range(len(keys)), key=lambda i: (-probabilities[i], keys[i]))
        pred_keys = [row["option_keys"][row["prediction"]] for row in rows]
        agreements += len(set(pred_keys)) == 1
        groups.append(base["split_group"])
        gold = base["gold"]
        per_golds.append(gold)
        for index, row in enumerate(rows):
            per_probs[index].append(aligned[index].tolist())
            per_corrects[index].append(int(pred_keys[index] == gold_key))
        records.append({key: base[key] for key in
                        ("qid", "split_group", "sample_id", "content_hash", "option_keys") if key in base} |
                       {"gold": gold, "prediction": prediction, "probabilities": probabilities.tolist(),
                        "max_probability": float(probabilities[prediction]), "correct": int(prediction == gold)})
    metrics = classification_metrics([row["probabilities"] for row in records], per_golds,
                                     predictions=[row["prediction"] for row in records])
    metrics.update(mean_conf=float(np.mean([row["max_probability"] for row in records])),
                   full_agreement_rate=agreements / len(records), predictions=records)
    corrects = [row["correct"] for row in records]
    variant_reports = []
    for index, (path, (_, data)) in enumerate(zip(paths, loaded)):
        variant_reports.append({"file": file_fingerprint(path), "n_file": len(sets[index]),
                                "excluded_from_common": len(sets[index] - common),
                                "excluded_item_hashes": sorted(sets[index] - common),
                                "metrics_on_common": classification_metrics(per_probs[index], per_golds),
                                "temperature": data.get("temperature"), "elapsed_s": data.get("elapsed_s"),
                                "paired_ensemble_delta": paired_delta(per_corrects[index], corrects, groups)})
    elapsed = sum(data.get("elapsed_s", 0) for _, data in loaded)
    metrics.update(elapsed_s=elapsed, forward_passes=sum(data.get("forward_passes", 0) for _, data in loaded),
                   latency_note="sum of component evaluations, including any order-diagnostic forwards")
    return {"protocol_version": 2, "method": method, "n_variants": len(paths), "n_common_items": len(common),
            "cohort_policy": "intersection" if allow_intersection else "identical",
            "per_variant": variant_reports, "ensemble": metrics,
            "confidence_quantiles": np.quantile([row["max_probability"] for row in records], [0, .25, .5, .75, 1]).tolist(),
            "prediction_positions": dict(Counter(row["prediction"] for row in records)),
            "gold_positions": dict(Counter(per_golds)),
            "calibration_note": "component probabilities averaged; no temperature or weights fitted on this evaluation",
            "selective_5pct": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--dev-evals", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=("probability_mean", "log_probability_mean"), default="probability_mean")
    ap.add_argument("--allow-intersection", action="store_true")
    args = ap.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    report = aggregate(args.evals, args.method, args.allow_intersection)
    if args.dev_evals:
        if len(args.dev_evals) != len(args.evals):
            raise ValueError("dev and evaluation must use the same number of components")
        dev = aggregate(args.dev_evals, args.method, args.allow_intersection)
        dev_rows, eval_rows = dev["ensemble"]["predictions"], report["ensemble"]["predictions"]
        if {row["split_group"] for row in dev_rows} & {row["split_group"] for row in eval_rows}:
            raise ValueError("acceptance dev and evaluation question groups overlap")
        policy = fit_acceptance_policy([row["max_probability"] for row in dev_rows], [row["correct"] for row in dev_rows])
        report["acceptance_policy"] = policy
        report["acceptance_dev_files"] = [file_fingerprint(path) for path in args.dev_evals]
        report["selective_5pct"] = evaluate_acceptance_policy(
            [row["max_probability"] for row in eval_rows], [row["correct"] for row in eval_rows], policy)
    with Path(args.out).open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report["ensemble"].items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
