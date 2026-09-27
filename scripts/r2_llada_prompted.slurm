#!/bin/bash
#SBATCH --job-name=r2_llada_prompted
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Round 2 (M10): prompted-generation baseline. LLaDA-8B-Instruct answers
# GPQA items by masked-diffusion generation (no trained head) so the paper
# can contrast "generate + parse" against "typed decision in one pass".

set -euo pipefail
RJ=/beegfs/a474r867/sciev
EXT=$RJ/data/bench_external
RUNS=$RJ/runs/sci
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

# use the CLEAN text mirrors (duplicate-option items excluded); build if absent
for CFG in main diamond; do
    CLEAN="$EXT/gpqa/gpqa_${CFG}_choice_eval_clean.jsonl"
    if [ ! -e "$CLEAN" ]; then
        "$PY" data/convert_benchmarks.py --bench gpqa \
            --src "$EXT/gpqa/dataset/gpqa_${CFG}.csv" --out "$CLEAN"
    fi
    DST="$RUNS/eval_llada_prompted_gpqa_${CFG}.json"
    if [ -e "$DST" ]; then echo "skip $DST"; continue; fi
    echo "=== prompted eval gpqa_$CFG ==="
    "$PY" scripts/eval_llada_prompted.py \
        --text-file "$EXT/gpqa/gpqa_${CFG}_choice_eval_clean_text.jsonl" \
        --out "$DST" --device cuda
done
echo "ALL_DONE"
