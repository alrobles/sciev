#!/bin/bash
#SBATCH --job-name=sci_llada5
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Phase B: retrain per-type heads on sci_battery_v2 (~30x data).
# Recipes = sci_llada4 winners + canonical order:
#   v2_choice : AttnPoolHead L{8,16,24,32}, canonical, 12000 steps
#   v2_noul   : mlp, canonical, 8000 steps
#   v2_score  : mlp, canonical, --ordinal 1.0, 8000 steps
# Eval: elite eval set (comparable) + v2 eval (in-distribution).

set -euo pipefail
RJ=/beegfs/a474r867/sciev
BAT=$RJ/data/sci_battery_v2
ELI=$RJ/data/sci_battery
RUNS=$RJ/runs/sci
PY=/beegfs/a474r867/bioai/venv/bin/python
LLADA=GSAI-ML/LLaDA-8B-Instruct
LAYERS="-1,-9,-17,-25"
mkdir -p "$RUNS"
LOCK=$RUNS/.sci_llada5.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another sci_llada5 holds the lock; exiting"
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

train() {  # tag head_kind steps kind extra...
    local TAG=$1 HK=$2 STEPS=$3 KIND=$4; shift 4
    [ -f "$RUNS/$TAG/decision.pt" ] && { echo "=== SKIP $TAG (done) ==="; return; }
    echo "=== TRAIN $TAG ==="
    $PY -m sciev.train \
        --decisions-train "$BAT/sci_${KIND}_train.jsonl" \
        --hf-backbone "$LLADA" \
        --r2-mode spanpool --head-kind "$HK" --r2-layers="$LAYERS" \
        --orders 1 --steps "$STEPS" --accum 1 --freeze \
        --canonical-order "$@" \
        --head-lr 3e-4 --warmup 200 --device cuda \
        --out "$RUNS/$TAG"
}

train v2_choice attnpool 12000 choice
train v2_noul   mlp      8000  noul
train v2_score  mlp      8000  score --ordinal 1.0

for KIND in choice noul score; do
    echo "=== EVAL elite $KIND (v2_$KIND) ==="
    $PY -m sciev.eval \
        --ckpt "$RUNS/v2_$KIND/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" \
        --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_${KIND}_dev.jsonl" \
        --decisions-eval "$ELI/sci_${KIND}_eval.jsonl" \
        --device cuda --out "$RUNS/eval_v2_${KIND}_elite.json"
    echo "=== EVAL v2 $KIND (v2_$KIND) ==="
    $PY -m sciev.eval \
        --ckpt "$RUNS/v2_$KIND/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" \
        --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_${KIND}_dev.jsonl" \
        --decisions-eval "$BAT/sci_${KIND}_eval.jsonl" \
        --device cuda --out "$RUNS/eval_v2_${KIND}_v2.json"
done
echo "ALL_DONE"
