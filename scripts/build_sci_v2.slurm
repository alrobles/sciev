#!/bin/bash
#SBATCH --job-name=sci_bat_v2
#SBATCH --partition=sixhour
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Phase B: sci_battery v2 — 25K train passages from qa_raw shards
# (polymer/physics + astronomy b2 + deriv), groundedness-filtered.
# Elite dev/eval untouched -> apples-to-apples with all previous numbers.

set -euo pipefail
RJ=/beegfs/a474r867/sciev
ECO=/beegfs/a474r867/ecoreasoner
PY=/beegfs/a474r867/bioai/venv/bin/python
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

$PY data/build_sci_decisions.py \
    --qa-raw "$ECO/data/qa_raw/qa_raw_shard*.jsonl" \
    --qa-raw "$ECO/data/qa_raw_b2/qa_raw_shard*.jsonl" \
    --qa-raw "$ECO/data/qa_raw_deriv/qa_raw_shard*.jsonl" \
    --qas-per-pid 2 \
    --train-pids 25000 --dev-pids 500 --eval-pids 500 \
    --out "$RJ/data/sci_battery_v2" \
    --seed 7331
echo DONE
