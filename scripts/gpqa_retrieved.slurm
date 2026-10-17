#!/bin/bash
#SBATCH --job-name=sciev_gpqa_retrieved
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

# Phase-B3: retrieved-evidence GPQA. Extends gpqa_grounded_v2 with a real
# retrieval arm: per question, BM25 top-k from the litdump PubMed FTS5
# index is appended to the state (same pairing contract: options/gold
# byte-identical to closed-book). Lands between nev (0.32) and oracle
# (0.70) and prices retrieval quality directly.

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
DB=${DB:-/beegfs/a474r867/litdump/pubmed/pubmed_full.db}
GPQA=$RJ/data/bench_external/gpqa
S2=$GPQA/systemone-v2
BAT=$RJ/data/sci_battery/systemone-v2
RUNS=$RJ/runs/sci
OUT=$RUNS/gpqa_grounded_v2
LAYERS=${LAYERS:--1,-9,-17,-25}
TOPK=${TOPK:-3}
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_GPQA_RETRIEVED job=%s code=%s db=%s\n' "$SLURM_JOB_ID" "$CODE" "$DB"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
[ -f "$DB" ] || { echo "retrieval db missing: $DB" >&2; exit 1; }
mkdir -p "$OUT"

# ---- build retrieved variants (CPU: FTS5 BM25 + tokenization) ----
for CFG in main diamond; do
    TARGET=$S2/gpqa_${CFG}_choice_eval_retrieved.jsonl
    if [ -f "$TARGET" ]; then
        printf 'skip existing %s\n' "$TARGET"
    else
        "$PY" "$CODE/data/build_gpqa_retrieved.py" \
            --text-file "$S2/gpqa_${CFG}_choice_eval_clean_text.jsonl" \
            --db "$DB" --top-k "$TOPK" --out "$TARGET"
    fi
done
printf 'BUILD_DONE\n'

# ---- eval: release head first, qa-v2 head second (non-fatal) ----
eval_one() {  # $1=ckpt-name $2=cfg
    local out="$OUT/eval_${1}_gpqa_${2}_retrieved.json"
    if [ -f "$out" ]; then
        printf 'skip existing %s\n' "$out"
        return 0
    fi
    "$PY" -m sciev.eval --ckpt "$RUNS/$1/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_choice_dev.jsonl" \
        --decisions-eval "$S2/gpqa_${2}_choice_eval_retrieved.jsonl" \
        --device cuda --out "$out"
}
for CFG in main diamond; do
    eval_one c_choice "$CFG"
done
printf 'C_CHOICE_DONE\n'
for CFG in main diamond; do
    eval_one qa_v2_s7331_choice "$CFG" || \
        printf 'qa_v2 eval failed for %s (non-fatal)\n' "$CFG" >&2
done
printf 'ALL_DONE\n'
