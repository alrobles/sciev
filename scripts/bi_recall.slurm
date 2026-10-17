#!/bin/bash
#SBATCH --job-name=sciev_bi_recall
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

# Phase-B1: can the bi-encoder shortlist the gold? Embeds banking77
# (3080 rows x 77 options, the K>4 test bed; unique sequences ~3.2k)
# into part p14, then measures recall@k / MRR of the trained heads:
#   pool=row   -> gold among the row's own options (the K>4 question)
#   pool=suite -> gold among every unique action in the file (retrieval)

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
QA=$RJ/data/qa_battery_v3/systemone-v2
BAT=$RJ/data/sci_battery/systemone-v2
GPQA=$RJ/data/bench_external/gpqa/systemone-v2
BANK=$RJ/data/bench_external/banking77
TAG=${TAG:-bi_v2_s7331}
OUT=$RJ/runs/sci/recall_${TAG}
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Invalid TAG\n' >&2; exit 1
fi
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_BI_RECALL job=%s tag=%s code=%s\n' "$SLURM_JOB_ID" "$TAG" "$CODE"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

HEADS=$RJ/runs/sci/${TAG}_biencoder/decision.pt
[ -f "$HEADS" ] || { printf 'missing heads %s\n' "$HEADS" >&2; exit 1; }

# banking77 is the only file needing fresh embeddings (part p14);
# resumable, so a second submission just skips ahead.
CACHE14=$RJ/runs/sci/${TAG}_embeds_p14.pt
"$PY" -m sciev.biencoder embed --backbone GSAI-ML/LLaDA-8B-Instruct \
    --decisions "$BANK/banking77_choice_eval.jsonl" --cache "$CACHE14" \
    --device cuda --batch 64 --save-every 20
printf 'EMBED_DONE part=14\n'

CACHES=()
for i in $(seq 0 14); do
    part=$RJ/runs/sci/${TAG}_embeds_p${i}.pt
    [ -f "$part" ] || { printf 'missing embed shard %s\n' "$part" >&2; exit 1; }
    CACHES+=(--cache "$part")
done
mkdir -p "$OUT"

# The K>4 prefilter question: gold label in the dot-product top-k.
"$PY" -m sciev.biencoder recall --heads "$HEADS" "${CACHES[@]}" \
    --decisions "$BANK/banking77_choice_eval.jsonl" --pool row \
    --ks 1 2 4 8 16 32 --out "$OUT/recall_banking77_row.json" --device cuda

# Same question on the 4-option suites (in-row ranking).
"$PY" -m sciev.biencoder recall --heads "$HEADS" "${CACHES[@]}" \
    --decisions "$QA/qa_choice_eval.jsonl" "$BAT/sci_choice_eval.jsonl" \
    "$GPQA/gpqa_main_choice_eval_clean.jsonl" \
    "$GPQA/gpqa_diamond_choice_eval_clean.jsonl" --pool row \
    --ks 1 2 4 --out "$OUT/recall_suites_row.json" --device cuda

# Global action retrieval: gold among every unique action of the suite.
"$PY" -m sciev.biencoder recall --heads "$HEADS" "${CACHES[@]}" \
    --decisions "$QA/qa_choice_eval.jsonl" "$BAT/sci_choice_eval.jsonl" \
    "$GPQA/gpqa_main_choice_eval_clean.jsonl" \
    "$GPQA/gpqa_diamond_choice_eval_clean.jsonl" --pool suite \
    --ks 1 4 16 64 --out "$OUT/recall_suites_global.json" --device cuda
printf 'ALL_DONE\n'
