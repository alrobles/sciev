#!/bin/bash
#SBATCH --job-name=sci_aligned_v2
#SBATCH --partition=sixhour
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Aligned systemone-v2 data build: elite battery + external benchmarks.
# Same source inputs as the historical c_* battery (qa_devin_elite.jsonl)
# but under the new contract: split-local negative pools, connected group
# splits, NFC-preserving identity, evidence spans, manifests with hashes.
# Outputs land in <out>/systemone-v2/ next to the legacy files.

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
ECO=${ECO:-/beegfs/a474r867/ecoreasoner}
BAT=${BAT:-$RJ/data/sci_battery}
EXT=${EXT:-$RJ/data/bench_external}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
SEED=${SEED:-7331}
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

$PY data/build_sci_decisions.py \
    --qa "$ECO/data/qa_devin_elite.jsonl" \
    --out "$BAT" --seed "$SEED" \
    --max-ctx 960 --overflow exclude

$PY data/convert_benchmarks.py --bench gpqa \
    --src "$EXT/gpqa/dataset/gpqa_main.csv" \
    --out "$EXT/gpqa/gpqa_main_choice_eval.jsonl" --seed "$SEED" \
    --max-ctx 960 --overflow exclude

$PY data/convert_benchmarks.py --bench gpqa \
    --src "$EXT/gpqa/dataset/gpqa_diamond.csv" \
    --out "$EXT/gpqa/gpqa_diamond_choice_eval.jsonl" --seed "$SEED" \
    --max-ctx 960 --overflow exclude

$PY data/convert_benchmarks.py --bench scifact \
    --src "$EXT/scifact/validation.parquet" \
    --out-prefix "$EXT/scifact/scifact_dev" --seed "$SEED" \
    --max-ctx 960 --overflow exclude

echo DONE
