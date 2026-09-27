#!/bin/bash
#SBATCH --job-name=r2_prompted_x
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Phase-A stronger generative baseline: LLaDA prompted with more diffusion
# steps and longer answer suffix (gen_len=16, steps=32) on clean GPQA.

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
EXT=${EXT:-$RJ/data/bench_external}
RUNS=${RUNS:-$RJ/runs/sci}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

for cfg in main diamond; do
    echo "=== PROMPTED-X gpqa_$cfg ==="
    "$PY" scripts/eval_llada_prompted.py \
        --text-file "$EXT/gpqa/systemone-v2/gpqa_${cfg}_choice_eval_text.jsonl" \
        --gen-len 16 --steps 32 --device cuda \
        --out "$RUNS/prompted_x_gpqa_${cfg}.json"
done
echo "ALL_DONE"
