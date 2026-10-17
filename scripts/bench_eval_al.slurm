#!/bin/bash
#SBATCH --job-name=bench_eval_al
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# External benchmarks on the antileak-retrained heads (c_*_al):
#   c_noul_al -> SciFact dev (noul)
#   c_score_al -> SciFact dev (score)
# GPQA is a choice task; no c_choice_al exists yet, so GPQA is not
# covered here. Temp-fit stays on each kind's elite dev (no benchmark
# dev leakage), same protocol as bench_eval.slurm.

set -euo pipefail
RJ=/beegfs/a474r867/sciev
BAT=$RJ/data/sci_battery/systemone-v2
EXT=$RJ/data/bench_external/scifact/systemone-v2
RUNS=$RJ/runs/sci
PY=/beegfs/a474r867/bioai/venv/bin/python
LAYERS="-1,-9,-17,-25"
mkdir -p "$RUNS"
LOCK=$RUNS/.bench_eval_al.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another bench_eval_al holds the lock; exiting"; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

# NOTE: legacy top-level files (sci_*_dev.jsonl, scifact_dev_*_eval.jsonl)
# fail the dataset contract (conflicting golds on identical effective
# inputs). Canonical evals use the systemone-v2 files, matching the
# fr_matched/r2_fr runs and the antileak_v2 evals.

echo "=== EVAL scifact noul (c_noul_al) ==="
$PY -m sciev.eval \
    --ckpt "$RUNS/c_noul_al/decision.pt" \
    --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
    --train-reference "$RUNS/antileak_v2/mix/sci_noul_al_train.jsonl" \
    --r2-temp-fit-decisions "$BAT/sci_noul_dev.jsonl" \
    --decisions-eval "$EXT/scifact_dev_noul_eval.jsonl" \
    --device cuda --out "$RUNS/eval_c_noul_al_scifact.json"

echo "=== EVAL scifact noul clean (c_noul_al) ==="
$PY -m sciev.eval \
    --ckpt "$RUNS/c_noul_al/decision.pt" \
    --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
    --train-reference "$RUNS/antileak_v2/mix/sci_noul_al_train.jsonl" \
    --r2-temp-fit-decisions "$BAT/sci_noul_dev.jsonl" \
    --decisions-eval "$EXT/clean_dev_noul_eval.jsonl" \
    --device cuda --out "$RUNS/eval_c_noul_al_scifact_clean.json"

echo "=== EVAL scifact score proxy (c_score_al) ==="
$PY -m sciev.eval \
    --ckpt "$RUNS/c_score_al/decision.pt" \
    --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
    --train-reference "$RUNS/antileak_v2/mix/sci_score_al_train.jsonl" \
    --r2-temp-fit-decisions "$BAT/sci_score_dev.jsonl" \
    --decisions-eval "$EXT/clean_dev_score_legacy_proxy_eval.jsonl" \
    --device cuda --out "$RUNS/eval_c_score_al_scifact.json"
echo "ALL_DONE"
