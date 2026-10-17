#!/bin/bash
#SBATCH --job-name=sciev_bi_v2
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

# CLM-style contrastive bi-encoder arm on the frozen LLaDA backbone:
# train the projection heads on cached vectors (shards produced by
# bi_embed.slurm), then evaluate under the same disjoint protocol v2
# as the joint head.

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
QA=$RJ/data/qa_battery_v3/systemone-v2
BAT=$RJ/data/sci_battery/systemone-v2
GPQA=$RJ/data/bench_external/gpqa/systemone-v2
TAG=${TAG:-bi_v2_s7331}
STEPS=${STEPS:-4000}
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Invalid TAG\n' >&2; exit 1
fi
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_BI_V2 job=%s tag=%s code=%s\n' "$SLURM_JOB_ID" "$TAG" "$CODE"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# Embedding shards produced by bi_embed.slurm (resumable per-file
# caches); submit this job with --dependency=afterok:<embed array>.
CACHES=()
for i in $(seq 0 13); do
    part=$RJ/runs/sci/${TAG}_embeds_p${i}.pt
    [ -f "$part" ] || { printf 'missing embed shard %s\n' "$part" >&2; exit 1; }
    CACHES+=(--cache "$part")
done
"$PY" -m sciev.biencoder train --decisions-train "$QA/qa_choice_train.jsonl" \
    --selection-dev "$QA/qa_choice_select.jsonl" \
    --heldout "$QA/qa_choice_calibration.jsonl" --heldout "$QA/qa_choice_policy.jsonl" \
    --heldout "$QA/qa_choice_eval.jsonl" --heldout "$BAT/sci_choice_eval.jsonl" \
    --heldout "$GPQA/gpqa_main_choice_eval_clean.jsonl" \
    --heldout "$GPQA/gpqa_diamond_choice_eval_clean.jsonl" \
    "${CACHES[@]}" --out "$RJ/runs/sci/${TAG}_biencoder" \
    --steps "$STEPS" --device cuda
"$PY" -m sciev.biencoder eval --heads "$RJ/runs/sci/${TAG}_biencoder/decision.pt" \
    "${CACHES[@]}" --calibration-dev "$QA/qa_choice_calibration.jsonl" \
    --policy-dev "$QA/qa_choice_policy.jsonl" \
    --evaluation "gpqa_main=$GPQA/gpqa_main_choice_eval_clean.jsonl" \
    --evaluation "gpqa_diamond=$GPQA/gpqa_diamond_choice_eval_clean.jsonl" \
    --evaluation "qa_eval=$QA/qa_choice_eval.jsonl" \
    --evaluation "sci_elite=$BAT/sci_choice_eval.jsonl" \
    --out-dir "$RJ/runs/sci/eval_${TAG}"
printf 'ALL_DONE\n'
