#!/bin/bash
#SBATCH --job-name=sciev_qa_eval_v2
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
TAG=${TAG:-qa_v2_s7331}
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Invalid TAG\n' >&2; exit 1
fi
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_QA_EVAL_V2 job=%s tag=%s\n' "$SLURM_JOB_ID" "$TAG"
"$PY" -m sciev.qa_eval --ckpt "$RJ/runs/sci/${TAG}_choice/decision.pt" \
    --qa-dir "$RJ/data/qa_battery_v3/systemone-v2" \
    --external-dir "$RJ/data/bench_external" --science-dir "$RJ/data/sci_battery/systemone-v2" \
    --out-dir "$RJ/runs/sci/eval_${TAG}"
printf 'ALL_DONE\n'
