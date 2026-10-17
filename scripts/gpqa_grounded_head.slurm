#!/bin/bash
#SBATCH --job-name=sciev_gpqa_head
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

# Evaluate an arbitrary choice head on every grounded GPQA arm that
# exists under systemone-v2 (clean|oracle|nev|retrieved x main|diamond).
# Outputs land in runs/sci/gpqa_grounded_v2/eval_<HEAD>_gpqa_<cfg>_<arm>.json
# with per-file skip guards (safe to resubmit).
#
# Usage:
#   sbatch --export=ALL,RJ=$RJ,HEAD=c_choice_alp25x \
#       scripts/gpqa_grounded_head.slurm

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
S2=$RJ/data/bench_external/gpqa/systemone-v2
BAT=$RJ/data/sci_battery/systemone-v2
RUNS=$RJ/runs/sci
OUT=$RUNS/gpqa_grounded_v2
HEAD=${HEAD:-c_choice_alp25x}
ARMS=${ARMS:-clean+oracle+nev+retrieved}
LAYERS=${LAYERS:--1,-9,-17,-25}
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_GPQA_HEAD job=%s head=%s arms=%s code=%s\n' \
    "$SLURM_JOB_ID" "$HEAD" "$ARMS" "$CODE"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
mkdir -p "$OUT"
[ -f "$RUNS/$HEAD/decision.pt" ] || { echo "missing ckpt $HEAD"; exit 1; }

eval_one() {  # $1=arm $2=cfg
    local ev="$S2/gpqa_${2}_choice_eval_${1}.jsonl"
    local out="$OUT/eval_${HEAD}_gpqa_${2}_${1}.json"
    [ -f "$ev" ] || { printf 'skip missing %s\n' "$ev"; return; }
    [ -f "$out" ] && { printf '=== SKIP %s ===\n' "$out"; return; }
    "$PY" -m sciev.eval --ckpt "$RUNS/$HEAD/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_choice_dev.jsonl" \
        --decisions-eval "$ev" --device cuda --out "$out"
}
IFS='+' read -ra ARMLIST <<< "$ARMS"
for CFG in main diamond; do
    for ARM in "${ARMLIST[@]}"; do
        eval_one "$ARM" "$CFG"
    done
done
printf 'ALL_DONE\n'
