#!/bin/bash
#SBATCH --job-name=sci_llada_ev
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.err
#
# Evals-only resume for sci_llada (both decision.pt already trained;
# job died on a bf16/fp32 dtype bug in eval_decisions_ids — now fixed).

set -euo pipefail
RJ=/beegfs/a474r867/reverse-jev
BAT=$RJ/data/sci_battery
RUNS=$RJ/runs/sci
PY=/beegfs/a474r867/bioai/venv/bin/python
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

cat "$BAT"/sci_choice_dev.jsonl \
    "$BAT"/sci_noul_dev.jsonl \
    "$BAT"/sci_score_dev.jsonl > "$BAT"/sci_all_dev.jsonl

for TAG in sci_llada_freeze sci_llada_ft; do
    for KIND in choice noul score; do
        echo "=== EVAL $KIND ($TAG) ==="
        $PY -m reverse_jev.eval \
            --ckpt "$RUNS/$TAG/decision.pt" \
            --r2-mode spanpool \
            --r2-temp-fit-decisions "$BAT/sci_all_dev.jsonl" \
            --decisions-eval "$BAT/sci_${KIND}_eval.jsonl" \
            --device cuda --out "$RUNS/eval_${TAG}_${KIND}.json"
    done
done
echo "ALL_DONE"
