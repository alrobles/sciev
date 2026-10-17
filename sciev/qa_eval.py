import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .calibration import fit_calibration, resolve_inference_settings, save_calibration
from .data import load_decisions_ids
from .eval import eval_decisions_ids
from .model import load_decision
from .protocol import assert_checkpoint_disjoint, assert_disjoint_splits, file_fingerprint
from .qa_train import require_training_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--qa-dir", required=True)
    ap.add_argument("--external-dir")
    ap.add_argument("--science-dir")
    ap.add_argument("--temperature-dev")
    ap.add_argument("--policy-dev")
    ap.add_argument("--evaluation", action="append", default=[])
    ap.add_argument("--r2-layers")
    ap.add_argument("--train-reference")
    ap.add_argument("--emit-policy-predictions", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7331)
    args = ap.parse_args()
    qa, out = Path(args.qa_dir), Path(args.out_dir)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    progress_path = Path(args.ckpt).parent / "progress.json"
    if progress_path.exists() and not json.loads(progress_path.read_text()).get("budget_complete"):
        raise ValueError("training budget is incomplete; resume before the final evaluation")
    layers = None if args.r2_layers is None else [int(value) for value in args.r2_layers.split(",")]
    inference = resolve_inference_settings(checkpoint, mode="spanpool", layers=layers,
                                           canonical=True, strict=True, decision_type="choice")
    temperature_path = Path(args.temperature_dev) if args.temperature_dev else qa / "qa_choice_calibration.jsonl"
    policy_path = Path(args.policy_dev) if args.policy_dev else qa / "qa_choice_policy.jsonl"
    temperature_rows, policy_rows = load_decisions_ids(temperature_path), load_decisions_ids(policy_path)
    require_training_data(temperature_rows)
    require_training_data(policy_rows)
    selection_checkpoint = {"training_data": checkpoint.get("selection_data")}
    for role, rows in (("temperature", temperature_rows), ("policy", policy_rows)):
        assert_checkpoint_disjoint(selection_checkpoint, rows, role=role)
    train_rows = load_decisions_ids(args.train_reference) if args.train_reference else None
    evaluations = {}
    for setting in args.evaluation:
        name, separator, path = setting.partition("=")
        if not separator or not name or Path(name).name != name or name in evaluations:
            ap.error("evaluation must be a unique name=path without directory separators")
        evaluations[name] = Path(path)
    if not evaluations:
        if not args.external_dir or not args.science_dir:
            ap.error("provide evaluations or both external-dir and science-dir")
        external = Path(args.external_dir)
        evaluations = {"gpqa_main": external / "gpqa/systemone-v2/gpqa_main_choice_eval_clean.jsonl",
                       "gpqa_diamond": external / "gpqa/systemone-v2/gpqa_diamond_choice_eval_clean.jsonl",
                       "qa_eval": qa / "qa_choice_eval.jsonl",
                       "sci_elite": Path(args.science_dir) / "sci_choice_eval.jsonl"}
    if "policy_predictions" in evaluations:
        ap.error("policy_predictions is reserved for acceptance-development predictions")
    if any((out / name).exists() for name in ["calibration.json", "policy_predictions.json", *(f"{name}.json" for name in evaluations)]):
        raise FileExistsError("evaluation artifacts already exist; use a fresh output directory")
    rows_by_name, checks = {}, {}
    for name, path in evaluations.items():
        rows = load_decisions_ids(path)
        if name.startswith("gpqa_main") and len(rows) != 440 or name.startswith("gpqa_diamond") and len(rows) != 193:
            raise ValueError(f"unexpected clean GPQA cohort size for {name}: {len(rows)}")
        roles = {"temperature": temperature_rows, "policy": policy_rows, "evaluation": rows}
        if train_rows is not None:
            roles["train_reference"] = train_rows
        assert_disjoint_splits(roles)
        check = assert_checkpoint_disjoint(checkpoint, rows)
        selection_check = assert_checkpoint_disjoint(selection_checkpoint, rows, role="model_selection")
        checks[name] = {key: value for key, value in check.items() if key != "data"}
        checks[name]["model_selection_check"] = {key: value for key, value in selection_check.items() if key != "data"}
        rows_by_name[name] = rows
    model, head = load_decision(args.ckpt, device=args.device)
    model.eval()
    head.eval()
    calibration = fit_calibration(model, head, temperature_rows, policy_rows,
                                  checkpoint_path=args.ckpt, device=args.device,
                                  mode="spanpool", layers=inference["layers"], canonical=True,
                                  strict=True, decision_type="choice", train_rows=train_rows)
    out.mkdir(parents=True, exist_ok=True)
    save_calibration(out / "calibration.json", calibration)
    if args.emit_policy_predictions:
        rows_by_name["policy_predictions"] = policy_rows
        evaluations["policy_predictions"] = policy_path
        check = assert_checkpoint_disjoint(checkpoint, policy_rows, role="acceptance_development")
        checks["policy_predictions"] = {key: value for key, value in check.items() if key != "data"}
    for name, rows in rows_by_name.items():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        metrics = eval_decisions_ids(model, head, rows, args.device, mode="spanpool",
                                     layers=inference["layers"], canonical=True, strict=True,
                                     temperature=calibration["temperature"], seed=args.seed, decision_type="choice",
                                     acceptance_policy=None if name == "policy_predictions" else calibration["acceptance_policy"],
                                     return_predictions=True)
        metrics.setdefault("selective", None)
        probabilities = [row["max_probability"] for row in metrics["predictions"]]
        metrics.update(confidence_quantiles=np.quantile(probabilities, [0, .25, .5, .75, 1]).tolist(),
                       mean_confidence=float(np.mean(probabilities)),
                       prediction_positions=dict(Counter(row["prediction"] for row in metrics["predictions"])),
                       gold_positions=dict(Counter(row["gold"] for row in metrics["predictions"])),
                       seconds_per_forward=metrics["elapsed_s"] / metrics["forward_passes"])
        report = {"protocol_version": 2,
                  "role": "acceptance_development" if name == "policy_predictions" else "evaluation",
                  "checkpoint": file_fingerprint(args.ckpt),
                  "input_file": file_fingerprint(evaluations[name]),
                  "temperature_file": file_fingerprint(temperature_path), "policy_file": file_fingerprint(policy_path),
                  "calibration_file": file_fingerprint(out / "calibration.json"),
                  "training_overlap": checks[name], "inference": inference,
                  "calibration_regime": "QA held-out temperature and acceptance; transferred unchanged to each test suite",
                  "code": [file_fingerprint(Path(__file__)), file_fingerprint(Path(__file__).with_name("eval.py"))],
                  "decisions_eval": {"file": str(evaluations[name]), **metrics}}
        with (out / f"{name}.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps({"suite": name, **{key: metrics[key] for key in
                                            ("n", "acc", "nll", "ece", "brier", "flip_rate", "selective", "elapsed_s")}}), flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
