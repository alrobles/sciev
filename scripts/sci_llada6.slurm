#!/bin/bash
#SBATCH --job-name=sci_llada6
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.err
#
# c_* recipe retrained on the DAPT-adapted backbone (LLaDA-8B + papers LoRA).
# Requires a trained adapter dir. Heads: ${TAG}_* (default da_*).
# Evals: GPQA main/diamond + SciFact noul/score + elite regression.
# Override adapter/tag without editing, e.g.:
#   sbatch --export=ALL,DAPT=$PWD/runs/dapt/lora-g2000,TAG=dag sci_llada6.slurm

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/reverse-jev}
BAT=${BAT:-$RJ/data/sci_battery/systemone-v2}
EXT=${EXT:-$RJ/data/bench_external}
RUNS=${RUNS:-$RJ/runs/sci}
DAPT=${DAPT:-$RJ/runs/dapt/lora-final}
SEED=${SEED:-7331}
TAG=${TAG:-da_matched_s${SEED}}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
LLADA=${LLADA:-GSAI-ML/LLaDA-8B-Instruct}
LAYERS=${LAYERS:--1,-9,-17,-25}
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "TAG must be a nonempty run name without path separators" >&2; exit 1
fi
ADAPTER_ARGS=()
if [ "$DAPT" != none ]; then
    if [ ! -f "$DAPT/adapter_config.json" ] || { [ ! -f "$DAPT/adapter_model.safetensors" ] && [ ! -f "$DAPT/adapter_model.bin" ]; }; then
        echo "missing adapter config or weights in $DAPT; use DAPT=none for the frozen arm" >&2; exit 1
    fi
    ADAPTER_ARGS=(--lora-adapter "$DAPT")
fi
reject_existing () {
    if [ -e "$1" ] || [ -L "$1" ]; then
        echo "refusing existing output $1; select a fresh TAG" >&2; exit 1
    fi
}
mkdir -p "$RUNS"
LOCK=$RUNS/.sci_llada6_${TAG}.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another sci_llada6 holds the lock; refusing this run" >&2; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

for K in choice noul score; do
    reject_existing "$RUNS/${TAG}_${K}"
done
for SUFFIX in choice_elite noul_elite score_elite choice_gpqa_main choice_gpqa_diamond noul_scifact choice_scifact; do
    reject_existing "$RUNS/eval_${TAG}_${SUFFIX}.json"
done

train_head () {  # name kind data extra...
    local name=$1 kind=$2 data=$3; shift 3
    reject_existing "$RUNS/$name"
    echo "=== TRAIN $name ==="
    "$PY" -m reverse_jev.train \
        --recipe scientific-v1 --decision-type "$kind" \
        --hf-backbone "$LLADA" "${ADAPTER_ARGS[@]}" \
        --decisions-train "$data" \
        --r2-layers="$LAYERS" --seed "$SEED" \
        --device cuda "$@" \
        --out "$RUNS/$name"
}

eval_head () {  # name evalfile out
    local name=$1 ev=$2 out=$3
    local kind=${name##*_}
    reject_existing "$RUNS/eval_${out}.json"
    echo "=== EVAL $out ($name) ==="
    "$PY" -m reverse_jev.eval \
        --ckpt "$RUNS/$name/decision.pt" \
        --decision-type "$kind" --seed "$SEED" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --train-reference "$BAT/sci_${kind}_train.jsonl" \
        --r2-temp-fit-decisions "$BAT/sci_${kind}_dev.jsonl" \
        --decisions-eval "$ev" \
        --device cuda --out "$RUNS/eval_${out}.json"
}

train_head "${TAG}_choice" choice "$BAT/sci_choice_train.jsonl"
train_head "${TAG}_noul"   noul   "$BAT/sci_noul_train.jsonl"
train_head "${TAG}_score"  score  "$BAT/sci_score_train.jsonl"

# elite regression (adapted backbone, same battery)
for K in choice noul score; do
    eval_head "${TAG}_$K" "$BAT/sci_${K}_eval.jsonl" "${TAG}_${K}_elite"
done
# external scientific benchmarks
eval_head "${TAG}_choice" "$EXT/gpqa/systemone-v2/gpqa_main_choice_eval.jsonl"    "${TAG}_choice_gpqa_main"
eval_head "${TAG}_choice" "$EXT/gpqa/systemone-v2/gpqa_diamond_choice_eval.jsonl" "${TAG}_choice_gpqa_diamond"
eval_head "${TAG}_noul"   "$EXT/scifact/systemone-v2/scifact_dev_noul_eval.jsonl" "${TAG}_noul_scifact"
eval_head "${TAG}_choice" "$EXT/scifact/systemone-v2/scifact_dev_choice_eval.jsonl" "${TAG}_choice_scifact"

echo "ALL_DONE"
