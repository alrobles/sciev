#!/bin/bash
#SBATCH --job-name=sciev_gpqa_grounded
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

# Phase-B2: grounded GPQA. Builds evidence-in-state variants paired with
# the closed-book clean files (same options/order, pid-shared):
#   oracle -> expert Explanation (gold-revealing upper bound)
#   nev    -> non-expert validator trace Explanation_NEV_* (noisy)
# then evaluates the release choice head (and the qa-v2 head) on each.

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
GPQA=$RJ/data/bench_external/gpqa
CSV=$GPQA/dataset/gpqa_main.csv
S2=$GPQA/systemone-v2
BAT=$RJ/data/sci_battery/systemone-v2
RUNS=$RJ/runs/sci
OUT=$RUNS/gpqa_grounded_v2
LAYERS=${LAYERS:--1,-9,-17,-25}
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_GPQA_GROUNDED job=%s code=%s\n' "$SLURM_JOB_ID" "$CODE"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
mkdir -p "$OUT"

# ---- build variants (CPU tokenization only) ----
for CFG in main diamond; do
    for ARM in oracle nev; do
        TARGET=$S2/gpqa_${CFG}_choice_eval_${ARM}.jsonl
        if [ -f "$TARGET" ]; then
            printf 'skip existing %s\n' "$TARGET"
        else
            "$PY" "$CODE/data/build_gpqa_grounded.py" --csv "$CSV" \
                --text-file "$S2/gpqa_${CFG}_choice_eval_clean_text.jsonl" \
                --arm "$ARM" --out "$TARGET"
        fi
    done
done
printf 'BUILD_DONE\n'

# ---- eval: release head first, qa-v2 head second ----
eval_one() {  # $1=ckpt-name $2=arm $3=cfg
    "$PY" -m sciev.eval --ckpt "$RUNS/$1/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_choice_dev.jsonl" \
        --decisions-eval "$S2/gpqa_${3}_choice_eval_${2}.jsonl" \
        --device cuda --out "$OUT/eval_${1}_gpqa_${3}_${2}.json"
}
for CFG in main diamond; do
    for ARM in oracle nev; do
        eval_one c_choice "$ARM" "$CFG"
    done
done
printf 'C_CHOICE_DONE\n'
for CFG in main diamond; do
    for ARM in oracle nev; do
        eval_one qa_v2_s7331_choice "$ARM" "$CFG" || \
            printf 'qa_v2 eval failed for %s %s (non-fatal)\n' "$ARM" "$CFG" >&2
    done
done
printf 'ALL_DONE\n'
