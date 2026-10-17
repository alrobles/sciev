#!/bin/bash
#SBATCH --job-name=sciev_antileak_choice
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

# Anti-leakage choice head: corrupted evidence has no reject class, so
# supervision comes in two forms (data/build_antileak_choice.py):
#   empty      -> uniform soft target (KL at --soft-weight 1.0)
#   transplant -> donor evidence spliced in; the alp25 noul head votes
#                 each option as "Proposed answer" -> confident support
#                 becomes the gold (evidence-following), otherwise the
#                 row gets a uniform soft target.
# Trains c_choice_${SUF} with the fr_choice recipe (attnpool, spanpool)
# plus embedded soft labels, then runs clean + evidence-control evals.
#
# Usage: sbatch --export=ALL,RJ=$RJ,SUF=alp25,FRAC=0.25 scripts/antileak_choice.slurm

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
BAT=$RJ/data/sci_battery/systemone-v2
BATH=$RJ/data/sci_battery_hard/systemone-v2
CTL=$RJ/data/sci_controls/systemone-v2
CTLH=$RJ/data/sci_controls_hard/systemone-v2
RUNS=$RJ/runs/sci
SUF=${SUF:?SUF required, e.g. alp25}
FRAC=${FRAC:-0.25}
DONOR=${DONOR:-near}
OUT=$RUNS/antileak_choice_${SUF}
LLADA=${LLADA:-GSAI-ML/LLaDA-8B-Instruct}
LAYERS=${LAYERS:--1,-9,-17,-25}
STEPS=${STEPS:-2000}
SEED=${SEED:-7331}
TAU=${TAU:-0.6}
MARGIN=${MARGIN:-0.2}
TEACHER_CKPT=${TEACHER_CKPT:-$RUNS/c_noul_${SUF}/decision.pt}
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_ANTILEAK_CHOICE job=%s code=%s frac=%s suf=%s teacher=%s\n' \
    "$SLURM_JOB_ID" "$CODE" "$FRAC" "$SUF" "$TEACHER_CKPT"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
mkdir -p "$OUT" "$OUT/mix/systemone-v2"

# ---- emit: empty controls + transplant candidates + teacher reqs ----
EMIT=$OUT/mix
if [ ! -f "$EMIT/systemone-v2/choice_empty_controls.jsonl" ] \
   || [ ! -f "$EMIT/teacher_reqs.jsonl" ]; then
    "$PY" "$CODE/data/build_antileak_choice.py" emit \
        --text "$BAT/sci_decisions_train_text.jsonl" \
        --control-frac "$FRAC" --donor "$DONOR" --seed "$SEED" \
        --controls-out "$EMIT/choice_empty_controls.jsonl" \
        --candidates "$EMIT/transplant_candidates.jsonl" \
        --teacher-out "$EMIT/teacher_reqs.jsonl"
fi
printf 'EMIT_DONE\n'

# ---- teacher eval: noul head votes each option as Proposed answer ----
if [ ! -f "$OUT/teacher_eval.json" ]; then
    "$PY" -m sciev.eval --ckpt "$TEACHER_CKPT" \
        --decision-type noul --seed "$SEED" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --decisions-eval "$EMIT/teacher_reqs.jsonl" --predictions \
        --device cuda --out "$OUT/teacher_eval.json"
fi
printf 'TEACHER_DONE\n'

# ---- relabel: confident support -> gold, else uniform soft ----
if [ ! -f "$EMIT/systemone-v2/choice_transplant_controls.jsonl" ]; then
    "$PY" "$CODE/data/build_antileak_choice.py" relabel \
        --candidates "$EMIT/transplant_candidates.jsonl" \
        --teacher-eval "$OUT/teacher_eval.json" \
        --tau "$TAU" --margin "$MARGIN" \
        --controls-out "$EMIT/choice_transplant_controls.jsonl" \
        --audit "$OUT/transplant_audit.jsonl"
fi
cat "$BAT/sci_choice_train.jsonl" \
    "$EMIT/systemone-v2/choice_empty_controls.jsonl" \
    "$EMIT/systemone-v2/choice_transplant_controls.jsonl" \
    > "$EMIT/sci_choice_${SUF}_train.jsonl"
printf 'MIX_DONE\n'

# ---- train: fr_choice recipe + embedded soft targets ----
TAG=c_choice_${SUF}
if [ ! -f "$RUNS/$TAG/decision.pt" ]; then
    "$PY" -m sciev.train \
        --decisions-train "$EMIT/sci_choice_${SUF}_train.jsonl" \
        --hf-backbone "$LLADA" --decision-type choice \
        --r2-mode spanpool --head-kind attnpool --r2-layers="$LAYERS" \
        --orders 1 --steps "$STEPS" --accum 1 --freeze \
        --canonical-order --seed "$SEED" \
        --soft-weight 1.0 --soft-temp 2.0 \
        --head-lr 3e-4 --warmup 200 --device cuda \
        --out "$RUNS/$TAG"
fi
printf 'TRAIN_DONE\n'

# ---- eval: clean + hard clean + empty/shuffle controls ----
eval_one() {  # $1=name $2=file
    local OUTJ="$OUT/eval_${TAG}_$1.json"
    [ -f "$OUTJ" ] && { printf '=== SKIP %s ===\n' "$OUTJ"; return; }
    "$PY" -m sciev.eval --ckpt "$RUNS/$TAG/decision.pt" \
        --decision-type choice --seed "$SEED" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --train-reference "$EMIT/sci_choice_${SUF}_train.jsonl" \
        --r2-temp-fit-decisions "$BAT/sci_choice_dev.jsonl" \
        --decisions-eval "$2" --device cuda \
        --out "$OUTJ"
}
eval_one clean "$BAT/sci_choice_eval.jsonl"
eval_one clean_hard "$BATH/sci_choice_eval.jsonl"
eval_one ctl_empty "$CTL/sci_controls_choice_empty_eval.jsonl"
eval_one ctl_shuffle "$CTL/sci_controls_choice_shuffle_eval.jsonl"
eval_one hard_ctl_empty "$CTLH/sci_controls_choice_empty_eval.jsonl"
eval_one hard_ctl_shuffle "$CTLH/sci_controls_choice_shuffle_eval.jsonl"
printf 'ALL_DONE\n'
