#!/bin/bash
#SBATCH --job-name=sciev_qa_v2
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
QA=$RJ/data/qa_battery_v3/systemone-v2
BAT=$RJ/data/sci_battery/systemone-v2
GPQA=$RJ/data/bench_external/gpqa/systemone-v2
TAG=${TAG:-qa_v2_s7331}
LAYERS=${LAYERS:--1,-9,-17,-25}
STEPS=${STEPS:-16000}
LORA_RANK=${LORA_RANK:-0}
SEED=${SEED:-7331}
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Invalid TAG\n' >&2; exit 1
fi
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_QA_V2 job=%s tag=%s lora_rank=%s code=%s\n' "$SLURM_JOB_ID" "$TAG" "$LORA_RANK" "$CODE"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
EXTRA=()
if [ "${SCI_MIX:-0}" = 1 ]; then EXTRA+=(--science-train "$BAT/sci_choice_train.jsonl" --science-repeats 15); fi
if [ -n "${HEAD_INIT:-}" ]; then EXTRA+=(--head-init "$HEAD_INIT"); fi
if [ -n "${RESUME:-}" ]; then EXTRA+=(--resume "$RESUME"); fi
if [ "${SMOKE:-0}" = 1 ]; then EXTRA+=(--smoke --steps 8 --save-every 4 --eval-every 4 --select-limit 8 --max-seconds 1800); fi
"$PY" -m sciev.qa_train --decisions-train "$QA/qa_choice_train.jsonl" \
    --selection-dev "$QA/qa_choice_select.jsonl" --r2-layers="$LAYERS" \
    --steps "$STEPS" --accum 4 --lora-rank "$LORA_RANK" --seed "$SEED" \
    --heldout "$QA/qa_choice_calibration.jsonl" --heldout "$QA/qa_choice_policy.jsonl" \
    --heldout "$QA/qa_choice_eval.jsonl" --heldout "$BAT/sci_choice_dev.jsonl" \
    --heldout "$BAT/sci_choice_eval.jsonl" --heldout "$GPQA/gpqa_main_choice_eval_clean.jsonl" \
    --heldout "$GPQA/gpqa_diamond_choice_eval_clean.jsonl" \
    --out "$RJ/runs/sci/${TAG}_choice" "${EXTRA[@]}"
printf 'ALL_DONE\n'
