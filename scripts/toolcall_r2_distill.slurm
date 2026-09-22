#!/bin/bash
#SBATCH --job-name=tc_distill
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/reverse-jev/runs/toolcall/%x_%j.out
#SBATCH --error=/beegfs/a474r867/reverse-jev/runs/toolcall/%x_%j.err

set -euo pipefail
RJ=/beegfs/a474r867/reverse-jev
ECO=/beegfs/a474r867/ecoreasoner
BAT=$RJ/data/toolcall_battery
RUNS=$RJ/runs/toolcall
SIF=$ECO/pytorch-cuda.sif
OVL=$ECO/blackwell-torch.overlay
PY=/bb/bwvenv/bin/python
mkdir -p "$RUNS"

CKPT=$ECO/runs/f2-spanes/checkpoint-g50000/model.pt
CFG=$ECO/harness/configs/f0-span-esqueleto.yaml
export HF_HOME=/beegfs/a474r867/hf-cache
cd "$RJ"

cat "$BAT"/toolcall_choice_train.jsonl \
    "$BAT"/toolcall_noul_train.jsonl \
    "$BAT"/toolcall_score_train.jsonl > "$BAT"/toolcall_all_train.jsonl
cat "$BAT"/toolcall_choice_dev.jsonl \
    "$BAT"/toolcall_noul_dev.jsonl \
    "$BAT"/toolcall_score_dev.jsonl > "$BAT"/toolcall_all_dev.jsonl

run() { apptainer exec --nv --overlay "$OVL":ro "$SIF" "$PY" "$@"; }

echo "=== TRAIN toolcall spanpool ft + OLMo-13B soft labels ==="
run -m reverse_jev.train \
    --decisions-train "$BAT/toolcall_all_train.jsonl" \
    --soft-labels "$RUNS/soft_train.jsonl" --soft-weight 0.4 --soft-temp 2.0 \
    --ckpt "$CKPT" --config "$CFG" \
    --r2-mode spanpool --steps 8000 --accum 1 \
    --lr 2e-5 --head-lr 3e-4 --warmup 200 --device cuda \
    --out "$RUNS/tc_sp_distill"

for KIND in choice noul score; do
    echo "=== EVAL $KIND (tc_sp_distill) ==="
    run -m reverse_jev.eval \
        --ckpt "$RUNS/tc_sp_distill/decision.pt" --config "$CFG" \
        --r2-mode spanpool \
        --r2-temp-fit-decisions "$BAT/toolcall_all_dev.jsonl" \
        --decisions-eval "$BAT/toolcall_${KIND}_eval.jsonl" \
        --device cuda --out "$RUNS/eval_tc_distill_${KIND}.json"
done
echo "ALL_DONE"
