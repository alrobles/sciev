#!/bin/bash
#SBATCH --job-name=sciev_gpqa_corpus
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

# Retrieved-evidence GPQA against an arbitrary litdump FTS5 db.
#   sbatch --export=ALL,DB=/beegfs/a474r867/litdump/wiki/wiki_full.db,TAG=wiki \
#          scripts/gpqa_retrieved_corpus.slurm
# Builds gpqa_main_choice_eval_retrieved_<TAG>.jsonl and evals the release
# + anti-leakage choice heads on it. Skip-on-existing everywhere.

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
: "${DB:?set DB=/path/to/fts5.db}"
: "${TAG:?set TAG=corpus name, e.g. wiki}"
S2=$RJ/data/bench_external/gpqa/systemone-v2
BAT=$RJ/data/sci_battery/systemone-v2
RUNS=$RJ/runs/sci
OUT=$RUNS/gpqa_grounded_v3
LAYERS=${LAYERS:--1,-9,-17,-25}
TOPK=${TOPK:-3}
MAXCTX=${MAXCTX:-1024}
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_GPQA_CORPUS job=%s db=%s tag=%s topk=%s maxctx=%s\n' \
    "$SLURM_JOB_ID" "$DB" "$TAG" "$TOPK" "$MAXCTX"
[ -f "$DB" ] || { echo "retrieval db missing: $DB" >&2; exit 1; }
mkdir -p "$OUT"

TARGET=$S2/gpqa_main_choice_eval_retrieved_${TAG}.jsonl
if [ -f "$TARGET" ]; then
    printf 'skip existing %s\n' "$TARGET"
else
    "$PY" "$CODE/data/build_gpqa_retrieved.py" \
        --text-file "$S2/gpqa_main_choice_eval_clean_text.jsonl" \
        --db "$DB" --top-k "$TOPK" --max-ctx "$MAXCTX" --out "$TARGET"
fi
printf 'BUILD_DONE\n'

eval_one() {  # $1=ckpt dir under $RUNS
    local out="$OUT/eval_${1}_gpqa_main_retrieved_${TAG}.json"
    if [ -f "$out" ]; then
        printf 'skip existing %s\n' "$out"
        return 0
    fi
    "$PY" -m sciev.eval --ckpt "$RUNS/$1/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_choice_dev.jsonl" \
        --decisions-eval "$TARGET" \
        --device cuda --out "$out"
}
for HEAD in c_choice c_choice_alp25x c_choice_alp25p; do
    eval_one "$HEAD" || printf '%s eval failed (non-fatal)\n' "$HEAD" >&2
done
printf 'ALL_DONE\n'
