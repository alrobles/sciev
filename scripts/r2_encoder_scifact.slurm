#!/bin/bash
#SBATCH --job-name=r2_encoder_scifact
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Round 2 (M10): trained encoder baseline on SciFact (DeBERTa-v3-base),
# evaluated on the same validation split as the decision heads.

set -euo pipefail
RJ=/beegfs/a474r867/sciev
EXT=$RJ/data/bench_external
RUNS=$RJ/runs/sci
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

"$PY" scripts/train_encoder_scifact.py \
    --train "$EXT/scifact/train.parquet" \
    --dev "$EXT/scifact/validation.parquet" \
    --out "$RUNS/eval_encoder_scifact.json" --device cuda
echo "ALL_DONE"
