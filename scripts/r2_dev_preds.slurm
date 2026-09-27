#!/bin/bash
#SBATCH --job-name=r2_dev_preds
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Round 2 (M7): per-item predictions on the DEV sets used for temperature
# fitting, so the selective-automation threshold can be fixed on dev and
# then applied once to the test predictions produced by the other jobs.

set -euo pipefail
RJ=/beegfs/a474r867/sciev
BAT=$RJ/data/sci_battery
EXT=$RJ/data/bench_external
RUNS=$RJ/runs/sci
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
LAYERS=${LAYERS:--1,-9,-17,-25}
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

for KIND in choice noul score; do
    DST="$RUNS/eval_r2_c_${KIND}_devpreds.json"
    if [ -e "$DST" ]; then echo "skip $DST"; continue; fi
    echo "=== DEV PREDS $KIND ==="
    "$PY" -m sciev.eval \
        --ckpt "$RUNS/c_${KIND}/decision.pt" \
        --decision-type "$KIND" --seed 7331 \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_${KIND}_dev.jsonl" \
        --decisions-eval "$BAT/sci_${KIND}_dev.jsonl" --predictions \
        --device cuda --out "$DST"
done

# test-set preds for the two public benchmarks with dev-selected threshold
for EV in "$EXT/scifact/scifact_dev_noul_eval.jsonl" \
          "$EXT/scifact/scifact_dev_score_eval.jsonl"; do
    KIND=$(basename "$EV" | sed 's/.*_\(noul\|score\)_eval.*/\1/')
    DST="$RUNS/eval_r2_c_${KIND}_scifact_preds.json"
    if [ -e "$DST" ]; then echo "skip $DST"; continue; fi
    echo "=== TEST PREDS scifact $KIND ==="
    "$PY" -m sciev.eval \
        --ckpt "$RUNS/c_${KIND}/decision.pt" \
        --decision-type "$KIND" --seed 7331 \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_${KIND}_dev.jsonl" \
        --decisions-eval "$EV" --predictions \
        --device cuda --out "$DST"
done
echo "ALL_DONE"
