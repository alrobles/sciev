#!/bin/bash
#SBATCH --job-name=sciev_bi_embed
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --array=0-13%4
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%A_%a.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%A_%a.err

# Per-file embedding shards for the CLM-style bi-encoder arm. One array
# task per decisions file -> one cache part; the cache saves
# incrementally and resumes, so a timed-out task is simply resubmitted
# (same command). Train/eval merge parts via repeated --cache args.

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
QA=$RJ/data/qa_battery_v3/systemone-v2
BAT=$RJ/data/sci_battery/systemone-v2
GPQA=$RJ/data/bench_external/gpqa/systemone-v2
TAG=${TAG:-bi_v2_s7331}
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Invalid TAG\n' >&2; exit 1
fi
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"

# The 50k-row train file does not fit one sixhour window at ~1 fwd/seq,
# so it is pre-sharded into runs/sci/embed_tmp/qa_train_shard{0..5}.jsonl
# (tasks 0 and 9-13); tasks 1-8 cover the remaining files.
TMPD=$RJ/runs/sci/embed_tmp
FILES=(
    "$TMPD/qa_train_shard0.jsonl" "$QA/qa_choice_select.jsonl"
    "$QA/qa_choice_calibration.jsonl" "$QA/qa_choice_policy.jsonl"
    "$QA/qa_choice_eval.jsonl"
    "$GPQA/gpqa_main_choice_eval_clean.jsonl"
    "$GPQA/gpqa_diamond_choice_eval_clean.jsonl"
    "$BAT/sci_choice_dev.jsonl" "$BAT/sci_choice_eval.jsonl"
    "$TMPD/qa_train_shard1.jsonl" "$TMPD/qa_train_shard2.jsonl"
    "$TMPD/qa_train_shard3.jsonl" "$TMPD/qa_train_shard4.jsonl"
    "$TMPD/qa_train_shard5.jsonl"
)
TARGET=${FILES[$SLURM_ARRAY_TASK_ID]}
CACHE=$RJ/runs/sci/${TAG}_embeds_p${SLURM_ARRAY_TASK_ID}.pt
printf 'TRACE_SCIEV_BI_EMBED job=%s task=%s file=%s code=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$TARGET" "$CODE"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
"$PY" -m sciev.biencoder embed --backbone GSAI-ML/LLaDA-8B-Instruct \
    --decisions "$TARGET" --cache "$CACHE" --device cuda --batch 64 \
    --save-every 20
printf 'EMBED_DONE part=%s\n' "$SLURM_ARRAY_TASK_ID"
