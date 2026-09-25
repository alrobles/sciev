#!/bin/bash
#SBATCH --job-name=dapt_llada_r
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.err
#
# Resume DAPT from lora-g2000 (job dapt_llada hit sixhour limit at ~3100/6000).
# Runs steps 2001..5000 (~4.8h at 5.7s/it) -> runs/dapt/lora-final.
# Original plan was 6000 steps but that never fit in one sixhour window;
# 5000 total (~82M tokens) is the release adapter.

set -euo pipefail
RJ=/beegfs/a474r867/reverse-jev
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
if [ ! -f "$RUNS/dapt/lora-g2000/adapter_config.json" ]; then
    echo "missing lora-g2000 — nothing to resume from"; exit 1
fi

$PY -m reverse_jev.dapt \
    --hf-backbone GSAI-ML/LLaDA-8B-Instruct \
    --corpus "$DATA/corpus_v4_en.jsonl" \
    --corpus "$DATA/corpus_v4hneg.jsonl" \
    --corpus "$DATA/corpus_v3_en.jsonl" \
    --corpus "$DATA/corpus_v5_pdb.jsonl" \
    --field text \
    --out "$RUNS/dapt" \
    --steps 5000 --seq-len 1024 --bs 16 --grad-accum 1 \
    --lr 2e-4 --lora-r 16 --lora-alpha 32 \
    --save-every 1000 \
    --resume "$RUNS/dapt/lora-g2000" --start-step 2000

echo "ALL_DONE"
