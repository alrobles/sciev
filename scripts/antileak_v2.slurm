#!/bin/bash
#SBATCH --job-name=sciev_antileak
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

# Phase-C2: anti-leakage arms for noul/score. Controls remap corrupted
# evidence to the reject label (empty/shuffle -> not-supported / lowest
# rubric), get mixed 1:1:1 with the clean train rows, and the heads are
# retrained with the same recipe as the release c_* heads. Success:
# agreement-with-reference on the control eval files drops toward ~0
# while the clean sci battery holds.

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
BAT=$RJ/data/sci_battery/systemone-v2
CTL=$RJ/data/sci_controls/systemone-v2
RUNS=$RJ/runs/sci
OUT=$RUNS/antileak_v2
LLADA=${LLADA:-GSAI-ML/LLaDA-8B-Instruct}
LAYERS=${LAYERS:--1,-9,-17,-25}
STEPS=${STEPS:-3000}
SEED=${SEED:-7331}
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_ANTILEAK job=%s code=%s\n' "$SLURM_JOB_ID" "$CODE"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
mkdir -p "$OUT" "$OUT/mix"

# ---- build remapped-gold controls + training mixes (CPU) ----
for KIND in noul score; do
    # write_dataset routes outputs through the encoding-version dir
    CTLS=$OUT/mix/systemone-v2/${KIND}_antileak_controls.jsonl
    if [ ! -f "$CTLS" ]; then
        "$PY" "$CODE/data/build_antileak_train.py" \
            --text "$BAT/sci_decisions_train_text.jsonl" \
            --kind "$KIND" --modes empty,shuffle --out "$CTLS"
    fi
    cat "$BAT/sci_${KIND}_train.jsonl" "$CTLS" \
        > "$OUT/mix/sci_${KIND}_al_train.jsonl"
done
printf 'BUILD_DONE\n'

# ---- train (same recipe as c_* heads: spanpool, frozen, canonical) ----
train() {  # $1=kind $2=extra
    local KIND=$1; shift
    local TAG=c_${KIND}_al
    [ -f "$RUNS/$TAG/decision.pt" ] && { printf '=== SKIP %s ===\n' "$TAG"; return; }
    "$PY" -m sciev.train \
        --decisions-train "$OUT/mix/sci_${KIND}_al_train.jsonl" \
        --hf-backbone "$LLADA" \
        --r2-mode spanpool --head-kind mlp --r2-layers="$LAYERS" \
        --orders 1 --steps "$STEPS" --accum 1 --freeze \
        --canonical-order "$@" \
        --head-lr 3e-4 --warmup 200 --device cuda \
        --out "$RUNS/$TAG"
}
train noul
train score --ordinal 1.0
printf 'TRAIN_DONE\n'

# ---- eval: clean battery + empty/shuffle controls per kind ----
eval_one() {  # $1=kind $2=name $3=file
    local OUTJ="$OUT/eval_c_$1_al_$2.json"
    [ -f "$OUTJ" ] && { printf '=== SKIP %s ===\n' "$OUTJ"; return; }
    "$PY" -m sciev.eval --ckpt "$RUNS/c_$1_al/decision.pt" \
        --decision-type "$1" --seed "$SEED" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --train-reference "$OUT/mix/sci_$1_al_train.jsonl" \
        --r2-temp-fit-decisions "$BAT/sci_$1_dev.jsonl" \
        --decisions-eval "$3" --device cuda \
        --out "$OUTJ"
}
for KIND in noul score; do
    eval_one "$KIND" clean "$BAT/sci_${KIND}_eval.jsonl"
    eval_one "$KIND" ctl_empty "$CTL/sci_controls_${KIND}_empty_eval.jsonl"
    eval_one "$KIND" ctl_shuffle "$CTL/sci_controls_${KIND}_shuffle_eval.jsonl"
done
printf 'ALL_DONE\n'
