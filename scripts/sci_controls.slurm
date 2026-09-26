#!/bin/bash
#SBATCH --job-name=sci_controls
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Evidence-control evals for one matched arm:
#   sbatch --export=ALL,ARM=fr_matched_s7331,DAPT=none sci_controls.slurm
#   sbatch --export=ALL,ARM=da_matched_s7331,DAPT=$PWD/runs/dapt/lora-final sci_controls.slurm
# Requires data/sci_controls/systemone-v2/ (build_evidence_controls.py output).
# Controls score agreement-with-reference, NOT accuracy on altered evidence.

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
BAT=${BAT:-$RJ/data/sci_battery/systemone-v2}
CTL=${CTL:-$RJ/data/sci_controls/systemone-v2}
RUNS=${RUNS:-$RJ/runs/sci}
ARM=${ARM:?ARM run tag required, e.g. fr_matched_s7331}
DAPT=${DAPT:-none}
SEED=${SEED:-7331}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
LLADA=${LLADA:-GSAI-ML/LLaDA-8B-Instruct}
LAYERS=${LAYERS:--1,-9,-17,-25}
if [[ ! "$ARM" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ARM must be a nonempty run name without path separators" >&2; exit 1
fi
if [ "$DAPT" != none ] && [ ! -d "$DAPT" ]; then
    echo "adapter dir $DAPT missing" >&2; exit 1
fi
for K in choice noul score; do
    [ -f "$RUNS/${ARM}_${K}/decision.pt" ] || { echo "missing head $RUNS/${ARM}_${K}/decision.pt"; exit 1; }
done
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

eval_ctl () {  # kind mode
    local kind=$1 mode=$2
    local out="$RUNS/eval_${ARM}_${kind}_ctl_${mode}.json"
    if [ -e "$out" ]; then echo "skip existing $out"; return; fi
    echo "=== EVAL ${ARM}_${kind}_ctl_${mode} ==="
    "$PY" -m sciev.eval \
        --ckpt "$RUNS/${ARM}_${kind}/decision.pt" \
        --decision-type "$kind" --seed "$SEED" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --train-reference "$BAT/sci_${kind}_train.jsonl" \
        --r2-temp-fit-decisions "$BAT/sci_${kind}_dev.jsonl" \
        --decisions-eval "$CTL/sci_controls_${kind}_${mode}_eval.jsonl" \
        --device cuda --out "$out"
}

for MODE in empty shuffle; do
    for K in choice noul score; do
        eval_ctl "$K" "$MODE"
    done
done

echo "ALL_DONE"
