#!/bin/bash
#SBATCH --job-name=bench_eval
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# External benchmarks on best elite heads (c_*):
#   c_choice -> GPQA main + diamond (choice)
#   c_noul   -> SciFact dev (noul)
#   c_score  -> SciFact dev (score)
# Temp-fit stays on each kind's elite dev (no benchmark dev leakage).

set -euo pipefail
RJ=/beegfs/a474r867/sciev
BAT=$RJ/data/sci_battery
EXT=$RJ/data/bench_external
RUNS=$RJ/runs/sci
PY=/beegfs/a474r867/bioai/venv/bin/python
LAYERS="-1,-9,-17,-25"
mkdir -p "$RUNS"
LOCK=$RUNS/.bench_eval.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another bench_eval holds the lock; exiting"; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

for CFG in main diamond; do
    echo "=== EVAL gpqa_$CFG (c_choice) ==="
    $PY -m sciev.eval \
        --ckpt "$RUNS/c_choice/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_choice_dev.jsonl" \
        --decisions-eval "$EXT/gpqa/gpqa_${CFG}_choice_eval.jsonl" \
        --device cuda --out "$RUNS/eval_c_choice_gpqa_${CFG}.json"
done
for KIND in noul score; do
    echo "=== EVAL scifact $KIND (c_$KIND) ==="
    $PY -m sciev.eval \
        --ckpt "$RUNS/c_$KIND/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_${KIND}_dev.jsonl" \
        --decisions-eval "$EXT/scifact/scifact_dev_${KIND}_eval.jsonl" \
        --device cuda --out "$RUNS/eval_c_${KIND}_scifact.json"
done
# Jev-contrast battery (published Jev numbers: Kumar 2026):
# enron 98.7 / sst2 95.7 / agnews 91.3 / banking77 76.0
for DS in enron/enron sst2/sst2 ag_news/agnews banking77/banking77; do
    echo "=== EVAL $DS (c_choice) ==="
    $PY -m sciev.eval \
        --ckpt "$RUNS/c_choice/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_choice_dev.jsonl" \
        --decisions-eval "$EXT/${DS}_choice_eval.jsonl" \
        --device cuda --out "$RUNS/eval_c_choice_$(basename $DS).json"
done
echo "ALL_DONE"
