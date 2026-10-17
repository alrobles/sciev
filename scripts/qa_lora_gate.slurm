#!/bin/bash
#SBATCH --job-name=sciev_lora_gate_v2
#SBATCH --partition=kbs
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_LORA_GATE_V2 job=%s\n' "$SLURM_JOB_ID"
"$PY" -m scripts.qa_lora_gate \
    --reports "$RJ/runs/sci/eval_qa_v2_s7331/gpqa_main.json" \
              "$RJ/runs/sci/eval_qamix_v2_s7331/gpqa_main.json" \
              "$RJ/runs/sci/eval_qa1l_v2_s7331/gpqa_main.json" \
    --data-audit "$RJ/runs/sci/gpqa_v3_data_audit.json" \
    --smoke-dir "$RJ/runs/sci/qa_lora_smoke_v2_choice" \
    --head-init "$RJ/runs/sci/qa_v2_s7331_choice/decision.pt" \
    --code "$CODE" --out "$RJ/runs/sci/qa_lora_gate_v2.json"
