#!/bin/bash
#SBATCH --job-name=qa_head
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Phase-B qa_choice head: fresh attnpool choice head on frozen LLaDA-8B,
# trained on the qa_choice battery (parametric MCQA). Evals: clean GPQA
# main/diamond + qa_choice held-out eval + elite choice regression.
# Arms:
#   sbatch --export=ALL,TRAIN=$QA/qa_choice_train.jsonl,TAG=qa_s7331 qa_head.slurm
#   sbatch --export=ALL,TRAIN=$QA/qasci_mix15_train.jsonl,TAG=qamix_s7331 qa_head.slurm

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
QA=${QA:-$RJ/data/qa_battery/systemone-v2}
BAT=${BAT:-$RJ/data/sci_battery/systemone-v2}
EXT=${EXT:-$RJ/data/bench_external}
RUNS=${RUNS:-$RJ/runs/sci}
SEED=${SEED:-7331}
TAG=${TAG:-qa_s${SEED}}
TRAIN=${TRAIN:-$QA/qa_choice_train.jsonl}
STEPS=${STEPS:-16000}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
LLADA=${LLADA:-GSAI-ML/LLaDA-8B-Instruct}
LAYERS=${LAYERS:--1,-9,-17,-25}
QADEV=${QADEV:-$QA/qa_choice_dev.jsonl}
SCIDEV=${SCIDEV:-$BAT/sci_choice_dev.jsonl}
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "TAG must be a nonempty run name without path separators" >&2; exit 1
fi
reject_existing () {
    if [ -e "$1" ] || [ -L "$1" ]; then
        echo "refusing existing output $1; select a fresh TAG" >&2; exit 1
    fi
}
mkdir -p "$RUNS"
LOCK=$RUNS/.qa_head_${TAG}.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another qa_head holds the lock; refusing this run" >&2; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

reject_existing "$RUNS/${TAG}_choice"
for S in gpqa_main gpqa_diamond qa_eval sci_elite; do
    reject_existing "$RUNS/eval_${TAG}_${S}.json"
done

echo "=== TRAIN ${TAG}_choice on $TRAIN (steps=$STEPS) ==="
"$PY" -m sciev.train \
    --recipe scientific-v1 --decision-type choice \
    --hf-backbone "$LLADA" \
    --decisions-train "$TRAIN" \
    --r2-layers="$LAYERS" --seed "$SEED" --steps "$STEPS" \
    --device cuda \
    --out "$RUNS/${TAG}_choice"

eval_head () {  # suffix evalfile tempfitdev
    local suf=$1 ev=$2 tdev=$3
    echo "=== EVAL ${TAG}_${suf} ==="
    "$PY" -m sciev.eval \
        --ckpt "$RUNS/${TAG}_choice/decision.pt" \
        --decision-type choice --seed "$SEED" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --train-reference "$TRAIN" \
        --r2-temp-fit-decisions "$tdev" \
        --decisions-eval "$ev" \
        --device cuda --predictions \
        --out "$RUNS/eval_${TAG}_${suf}.json"
}

eval_head gpqa_main    "$EXT/gpqa/systemone-v2/gpqa_main_choice_eval.jsonl"    "$QADEV"
eval_head gpqa_diamond "$EXT/gpqa/systemone-v2/gpqa_diamond_choice_eval.jsonl" "$QADEV"
eval_head qa_eval      "$QA/qa_choice_eval_sub2k.jsonl"                        "$QADEV"
eval_head sci_elite    "$BAT/sci_choice_eval.jsonl"                            "$SCIDEV"

echo "ALL_DONE"
