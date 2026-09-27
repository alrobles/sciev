#!/bin/bash
#SBATCH --job-name=r2_ensemble
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Phase-A prompt ensemble: eval c_choice on GPQA instruction variants
# (v1..v4) with per-item predictions; baseline variant v0 = existing clean
# eval. Aggregation offline by sample_id.

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
BAT=${BAT:-$RJ/data/sci_battery/systemone-v2}
EXT=${EXT:-$RJ/data/bench_external}
RUNS=${RUNS:-$RJ/runs/sci}
SEED=${SEED:-7331}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
LAYERS=${LAYERS:--1,-9,-17,-25}
CKPT=${CKPT:-$RUNS/c_choice/decision.pt}
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

eval_one () {  # tag evalfile
    local tag=$1 ev=$2
    echo "=== EVAL $tag ==="
    "$PY" -m sciev.eval \
        --ckpt "$CKPT" \
        --decision-type choice --seed "$SEED" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --train-reference "$BAT/sci_choice_train.jsonl" \
        --r2-temp-fit-decisions "$BAT/sci_choice_dev.jsonl" \
        --decisions-eval "$ev" \
        --device cuda --predictions \
        --out "$RUNS/eval_${tag}.json"
}

for cfg in main diamond; do
    for vi in 1 2 3 4; do
        eval_one "c_choice_gpqa_${cfg}_instr${vi}" \
            "$EXT/gpqa/systemone-v2/gpqa_${cfg}_choice_eval_instr${vi}.jsonl"
    done
done
echo "ALL_DONE"
