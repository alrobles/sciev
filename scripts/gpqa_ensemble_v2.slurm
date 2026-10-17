#!/bin/bash
#SBATCH --job-name=sciev_ensemble_v2
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-4%1
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%A_%a.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%A_%a.err

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
OUT=${OUT:-$RJ/runs/sci/phase_a_v2}
QA=$RJ/data/qa_battery_v2/systemone-v2
VAR=$OUT/instructions/systemone-v2
VI=$SLURM_ARRAY_TASK_ID
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_ENSEMBLE_V2 job=%s variant=%s\n' "$SLURM_JOB_ID" "$VI"
"$PY" -m sciev.qa_eval --ckpt "$RJ/runs/sci/c_choice/decision.pt" \
    --qa-dir "$QA" --temperature-dev "$VAR/calibration_instr${VI}.jsonl" \
    --policy-dev "$VAR/policy_instr${VI}.jsonl" --emit-policy-predictions \
    --r2-layers=-1,-9,-17,-25 \
    --evaluation "gpqa_main=$VAR/main_instr${VI}.jsonl" \
    --evaluation "gpqa_diamond=$VAR/diamond_instr${VI}.jsonl" \
    --out-dir "$OUT/ensemble/v${VI}"
printf 'ALL_DONE\n'
