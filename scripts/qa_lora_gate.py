# Predefined LoRA gate for the qa-v2 line: fires a rank-8 LoRA arm only
# when every verified frozen-head GPQA-main report lands below 0.38.
#
# POST-PHASE-A NOTE (2026-10): the gate's premise — "if heads underperform,
# adaptation may unlock latent backbone knowledge" — is superseded by the
# backbone-only diagnostics in runs/sci/phase_a_v2/. No head-free protocol
# (PLL variants, prompted generation) beats the trained head, so GPQA is a
# backbone knowledge ceiling, not a readout bottleneck. Do NOT fire this
# gate on MCQA data expecting GPQA gains; the only coherent reformulation
# is DAPT over a scientific text corpus (knowledge injection), whose
# expected ceiling on retrieval-resistant questions remains low.
import argparse
import json
import os
import subprocess
from pathlib import Path

from sciev.protocol import file_fingerprint


def decide(reports, expected_sha, threshold=0.38):
    if not reports or not 0 < threshold < 1:
        raise ValueError("require reports and a threshold between zero and one")
    scores = []
    for report in reports:
        metrics = report.get("decisions_eval", {})
        if (report.get("protocol_version") != 2 or report.get("role") != "evaluation"
                or report.get("input_file", {}).get("sha256") != expected_sha
                or metrics.get("n") != 440 or len(metrics.get("predictions", [])) != 440
                or report.get("training_overlap", {}).get("status") != "verified"):
            raise ValueError("LoRA gate requires complete, verified evaluations on the fixed clean GPQA cohort")
        accuracy = metrics.get("acc")
        if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not 0 <= accuracy <= 1:
            raise ValueError("invalid accuracy in gate input")
        predictions = metrics["predictions"]
        if any(type(row.get(key)) is not int or not 0 <= row[key] < 4
               for row in predictions for key in ("prediction", "gold")):
            raise ValueError("invalid per-item predictions in gate input")
        measured = sum(row["prediction"] == row["gold"] for row in predictions) / 440
        if abs(measured - accuracy) > 1e-10:
            raise ValueError("reported accuracy disagrees with per-item predictions")
        scores.append(float(accuracy))
    return {"launch": max(scores) < threshold, "threshold": threshold, "accuracies": scores,
            "criterion": "all predefined frozen-head GPQA-main accuracies below threshold",
            "interpretation": "benchmark-informed experiment gate, not a test of a backbone knowledge ceiling"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--data-audit", required=True)
    ap.add_argument("--smoke-dir", required=True)
    ap.add_argument("--head-init", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--tag", default="qalora_v2_s7331")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.38)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise FileExistsError("gate output already exists; inspect submitted job IDs before retrying")
    audit = json.loads(Path(args.data_audit).read_text())
    reports = [json.loads(Path(path).read_text()) for path in args.reports]
    result = decide(reports, audit["inputs"]["gpqa_main"]["sha256"], args.threshold)
    result["input_reports"] = [file_fingerprint(path) for path in args.reports]
    result["initial_head"] = file_fingerprint(args.head_init)
    with Path(args.out).open("x") as stream:
        json.dump({**result, "status": "gate_started_no_jobs_submitted"}, stream, indent=2)
    if result["launch"]:
        smoke_dir = Path(args.smoke_dir)
        smoke = json.loads((smoke_dir / "progress.json").read_text())
        smoke_config = json.loads((smoke_dir / "training_config.json").read_text())
        if (not smoke.get("budget_complete") or not smoke.get("reload_verified")
                or smoke.get("changed_adapter_tensors", 0) < 1 or smoke_config.get("lora_rank") != 8):
            raise ValueError("rank-8 LoRA GPU smoke did not pass; no training job submitted")
        code = Path(args.code)
        environment = {key: value for key, value in os.environ.items() if key not in
                       ("TAG", "HEAD_INIT", "RESUME", "SMOKE", "SCI_MIX", "STEPS", "LAYERS", "LORA_RANK")}
        export = f"ALL,CODE={code},TAG={args.tag},LORA_RANK=8,STEPS=4000,SCI_MIX=0,SMOKE=0,HEAD_INIT={args.head_init}"
        train_job = subprocess.run(["sbatch", "--parsable", f"--export={export}", str(code / "scripts/qa_v2_train.slurm")],
                                   check=True, capture_output=True, text=True, env=environment).stdout.strip().split(";")[0]
        if not train_job.isdigit():
            raise ValueError(f"unexpected sbatch response {train_job!r}")
        result.update(status="training_submitted", training_job=train_job)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        eval_job = subprocess.run(["sbatch", "--parsable", f"--dependency=afterok:{train_job}",
                                  f"--export=ALL,CODE={code},TAG={args.tag}", str(code / "scripts/qa_v2_eval.slurm")],
                                 check=True, capture_output=True, text=True, env=environment).stdout.strip().split(";")[0]
        if not eval_job.isdigit():
            raise ValueError(f"unexpected sbatch response {eval_job!r}")
        result.update(status="training_and_evaluation_submitted", evaluation_job=eval_job, smoke=smoke)
    else:
        result["status"] = "skipped_threshold_reached"
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
