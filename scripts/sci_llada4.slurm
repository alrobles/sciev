#!/bin/bash
#SBATCH --job-name=sci_llada4
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Phase A: canonical option order (flip=0 by construction) + ordinal head.
# Per-type heads on frozen LLaDA-8B, best recipe per kind:
#   c_choice : AttnPoolHead L{8,16,24,32}, canonical, orders=1
#   c_noul   : DecisionHead (mlp), canonical, orders=1
#   c_score  : DecisionHead (mlp), canonical, --ordinal 1.0
# NB: canonical makes --orders redundant (all draws identical).
# Evals must pass --canonical-order to match.

set -euo pipefail
RJ=/beegfs/a474r867/sciev
BAT=$RJ/data/sci_battery
RUNS=$RJ/runs/sci
PY=/beegfs/a474r867/bioai/venv/bin/python
LLADA=GSAI-ML/LLaDA-8B-Instruct
LAYERS="-1,-9,-17,-25"
mkdir -p "$RUNS"
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

train() {  # tag head_kind orders steps kind extra_args...
    local TAG=$1 HK=$2 ORD=$3 STEPS=$4 KIND=$5; shift 5
    [ -f "$RUNS/$TAG/decision.pt" ] && { echo "=== SKIP $TAG (done) ==="; return; }
    echo "=== TRAIN $TAG ==="
    $PY -m sciev.train \
        --decisions-train "$BAT/sci_${KIND}_train.jsonl" \
        --hf-backbone "$LLADA" \
        --r2-mode spanpool --head-kind "$HK" --r2-layers="$LAYERS" \
        --orders "$ORD" --steps "$STEPS" --accum 1 --freeze \
        --canonical-order "$@" \
        --head-lr 3e-4 --warmup 200 --device cuda \
        --out "$RUNS/$TAG"
}

train c_choice attnpool 1 2000 choice
train c_noul   mlp      1 3000 noul
train c_score  mlp      1 3000 score --ordinal 1.0

for KIND in choice noul score; do
    echo "=== EVAL $KIND (c_$KIND) ==="
    $PY -m sciev.eval \
        --ckpt "$RUNS/c_$KIND/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" \
        --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_${KIND}_dev.jsonl" \
        --decisions-eval "$BAT/sci_${KIND}_eval.jsonl" \
        --device cuda --out "$RUNS/eval_c_${KIND}.json"
done
echo "ALL_DONE"
