#!/bin/bash
#SBATCH --job-name=dapt_llada
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Domain-adaptive pretraining of LLaDA-8B on the papers corpus.
# Native masked-diffusion objective (mask ~ U(0,1), CE/t), LoRA r=16.
# 6000 steps x bs16 x 1024tok ~ 96K docs ~ 100M tokens.
# Output: runs/dapt/lora-final (adapter, ~300MB) — merged at load time
# via HFBackbone(lora_adapter=...).

set -euo pipefail
RJ=/beegfs/a474r867/sciev
RUNS=$RJ/runs
DATA=/beegfs/a474r867/ecoreasoner/data
PY=/beegfs/a474r867/bioai/venv/bin/python
mkdir -p "$RUNS/dapt"
LOCK=$RUNS/.dapt_llada.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another dapt_llada holds the lock; exiting"; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

if [ -f "$RUNS/dapt/lora-final/adapter_config.json" ]; then
    echo "=== SKIP dapt (lora-final exists) ==="; exit 0
fi

$PY -m sciev.dapt \
    --hf-backbone GSAI-ML/LLaDA-8B-Instruct \
    --corpus "$DATA/corpus_v4_en.jsonl" \
    --corpus "$DATA/corpus_v4hneg.jsonl" \
    --corpus "$DATA/corpus_v3_en.jsonl" \
    --corpus "$DATA/corpus_v5_pdb.jsonl" \
    --field text \
    --out "$RUNS/dapt" \
    --steps 6000 --seq-len 1024 --bs 16 --grad-accum 1 \
    --lr 2e-4 --lora-r 16 --lora-alpha 32 \
    --save-every 2000

echo "ALL_DONE"
