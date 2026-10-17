#!/bin/bash
#SBATCH --job-name=sciev_antileak_frac
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

# Phase-C3: anti-leakage arms with a rebalanced control mix. v2 mixed
# controls 1:1:1 vs clean rows, which collapsed the label prior toward
# the reject class (score: 78% level-0; noul: 89% false) and cost clean
# accuracy. Here --control-frac subsamples the control rows per mode so
# the reject prior stays closer to the eval prior while still teaching
# evidence conditioning.
#
# Usage: sbatch --export=ALL,RJ=$RJ,FRAC=0.25,SUF=alp25 scripts/antileak_frac.slurm
#   FRAC: fraction of each control mode kept (0<f<=1)
#   SUF:  run tag -> heads c_{kind}_${SUF}, outputs under runs/sci/antileak_${SUF}

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
BAT=$RJ/data/sci_battery/systemone-v2
CTL=$RJ/data/sci_controls/systemone-v2
EXT=$RJ/data/bench_external/scifact/systemone-v2
RUNS=$RJ/runs/sci
SUF=${SUF:?SUF required, e.g. alp25}
FRAC=${FRAC:?FRAC required, e.g. 0.25}
OUT=$RUNS/antileak_${SUF}
LLADA=${LLADA:-GSAI-ML/LLaDA-8B-Instruct}
LAYERS=${LAYERS:--1,-9,-17,-25}
STEPS=${STEPS:-3000}
SEED=${SEED:-7331}
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_ANTILEAK job=%s code=%s frac=%s suf=%s\n' "$SLURM_JOB_ID" "$CODE" "$FRAC" "$SUF"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
mkdir -p "$OUT" "$OUT/mix"

# ---- build subsampled remapped-gold controls + training mixes (CPU) ----
for KIND in noul score; do
    # write_dataset routes outputs through the encoding-version dir
    CTLS=$OUT/mix/systemone-v2/${KIND}_antileak_controls.jsonl
    if [ ! -f "$CTLS" ]; then
        "$PY" "$CODE/data/build_antileak_train.py" \
            --text "$BAT/sci_decisions_train_text.jsonl" \
            --kind "$KIND" --modes empty,shuffle \
            --control-frac "$FRAC" --out "$CTLS"
    fi
    cat "$BAT/sci_${KIND}_train.jsonl" "$CTLS" \
        > "$OUT/mix/sci_${KIND}_${SUF}_train.jsonl"
done
printf 'BUILD_DONE\n'

# ---- train (same recipe as c_* heads: spanpool, frozen, canonical) ----
train() {  # $1=kind $2=extra
    local KIND=$1; shift
    local TAG=c_${KIND}_${SUF}
    [ -f "$RUNS/$TAG/decision.pt" ] && { printf '=== SKIP %s ===\n' "$TAG"; return; }
    "$PY" -m sciev.train \
        --decisions-train "$OUT/mix/sci_${KIND}_${SUF}_train.jsonl" \
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

# ---- eval: clean battery + empty/shuffle controls + scifact per kind ----
eval_one() {  # $1=kind $2=name $3=file
    local OUTJ="$OUT/eval_c_$1_${SUF}_$2.json"
    [ -f "$OUTJ" ] && { printf '=== SKIP %s ===\n' "$OUTJ"; return; }
    "$PY" -m sciev.eval --ckpt "$RUNS/c_$1_${SUF}/decision.pt" \
        --decision-type "$1" --seed "$SEED" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --train-reference "$OUT/mix/sci_$1_${SUF}_train.jsonl" \
        --r2-temp-fit-decisions "$BAT/sci_$1_dev.jsonl" \
        --decisions-eval "$3" --device cuda \
        --out "$OUTJ"
}
for KIND in noul score; do
    eval_one "$KIND" clean "$BAT/sci_${KIND}_eval.jsonl"
    eval_one "$KIND" ctl_empty "$CTL/sci_controls_${KIND}_empty_eval.jsonl"
    eval_one "$KIND" ctl_shuffle "$CTL/sci_controls_${KIND}_shuffle_eval.jsonl"
done
eval_one noul scifact "$EXT/scifact_dev_noul_eval.jsonl"
eval_one noul scifact_clean "$EXT/clean_dev_noul_eval.jsonl"
eval_one score scifact "$EXT/clean_dev_score_legacy_proxy_eval.jsonl"
printf 'ALL_DONE\n'
