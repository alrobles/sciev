#!/bin/bash
#SBATCH --job-name=sci_hard_eval
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Hard-distractor choice evals + evidence controls for one matched arm:
#   sbatch --export=ALL,ARM=fr_matched_s7331,DAPT=none sci_hard_eval.slurm
#   sbatch --export=ALL,ARM=da_matched_s7331,DAPT=$PWD/runs/dapt/lora-final sci_hard_eval.slurm

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
BAT=${BAT:-$RJ/data/sci_battery/systemone-v2}
HARD=${HARD:-$RJ/data/sci_battery_hard/systemone-v2}
HCTL=${HCTL:-$RJ/data/sci_controls_hard/systemone-v2}
RUNS=${RUNS:-$RJ/runs/sci}
ARM=${ARM:?ARM run tag required}
DAPT=${DAPT:-none}
SEED=${SEED:-7331}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
LAYERS=${LAYERS:--1,-9,-17,-25}
[ -f "$RUNS/${ARM}_choice/decision.pt" ] || { echo "missing $RUNS/${ARM}_choice/decision.pt"; exit 1; }
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

run_eval () {  # evalfile out
    local ev=$1 out=$2
    local dst="$RUNS/eval_${out}.json"
    if [ -e "$dst" ]; then echo "skip existing $dst"; return; fi
    echo "=== EVAL $out ==="
    "$PY" -m sciev.eval \
        --ckpt "$RUNS/${ARM}_choice/decision.pt" \
        --decision-type choice --seed "$SEED" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --train-reference "$BAT/sci_choice_train.jsonl" \
        --r2-temp-fit-decisions "$BAT/sci_choice_dev.jsonl" \
        --decisions-eval "$ev" \
        --device cuda --out "$dst"
}

run_eval "$HARD/sci_choice_eval.jsonl"                          "${ARM}_choice_hard"
run_eval "$HCTL/sci_controls_choice_empty_eval.jsonl"           "${ARM}_choice_hard_ctl_empty"
run_eval "$HCTL/sci_controls_choice_shuffle_eval.jsonl"         "${ARM}_choice_hard_ctl_shuffle"

echo "ALL_DONE"
