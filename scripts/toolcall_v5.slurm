#!/bin/bash
#SBATCH --job-name=tc_v5
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/reverse-jev/runs/toolcall/%x_%j.out
#SBATCH --error=/beegfs/a474r867/reverse-jev/runs/toolcall/%x_%j.err
#
# Spanpool-FT on the tool-call battery initialized from the v5-1b
# backbone (MoE 768/12L/12H/16E top-1, ~1.13B) — the capacity test.
# Submit with:  sbatch --dependency=afterany:<v5-1b jobids> scripts/toolcall_v5.slurm

set -euo pipefail
RJ=/beegfs/a474r867/reverse-jev
ECO=/beegfs/a474r867/ecoreasoner
BAT=$RJ/data/toolcall_battery
RUNS=$RJ/runs/toolcall
SIF=$ECO/pytorch-cuda.sif
OVL=$ECO/blackwell-torch.overlay
PY=/bb/bwvenv/bin/python
CFG=$ECO/harness/configs/eval-moe-v5.yaml
mkdir -p "$RUNS"
export HF_HOME=/beegfs/a474r867/hf-cache
cd "$RJ"

# --- pick latest *complete* v5-1b checkpoint (size > 4GB, else previous) ---
CKPT=""
for d in $(ls -d $ECO/runs/v5-1b/checkpoint-g*/ 2>/dev/null | sort -t g -k2 -n -r); do
    f="${d}model.pt"
    if [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 4000000000 ]; then
        CKPT="$f"; break
    fi
done
[ -n "$CKPT" ] || { echo "no complete v5-1b checkpoint found"; exit 1; }
echo "=== v5-1b checkpoint: $CKPT ==="

run() { apptainer exec --nv --overlay "$OVL":ro "$SIF" "$PY" "$@"; }

echo "=== TRAIN toolcall spanpool ft (v5-1b init) ==="
run -m reverse_jev.train \
    --decisions-train "$BAT/toolcall_all_train.jsonl" \
    --ckpt "$CKPT" --config "$CFG" \
    --r2-mode spanpool --steps 8000 --accum 1 \
    --lr 2e-5 --head-lr 3e-4 --warmup 200 --device cuda \
    --out "$RUNS/tc_v5_sp"

for KIND in choice noul score; do
    echo "=== EVAL $KIND (tc_v5_sp) ==="
    run -m reverse_jev.eval \
        --ckpt "$RUNS/tc_v5_sp/decision.pt" --config "$CFG" \
        --r2-mode spanpool \
        --r2-temp-fit-decisions "$BAT/toolcall_all_dev.jsonl" \
        --decisions-eval "$BAT/toolcall_${KIND}_eval.jsonl" \
        --device cuda --out "$RUNS/eval_tc_v5_${KIND}.json"
done

echo "=== EVAL pairs v4 holdout (comparability with 155M) ==="
run -m reverse_jev.eval \
    --ckpt "$RUNS/tc_v5_sp/decision.pt" --config "$CFG" \
    --r2-mode spanpool \
    --pairs-dir "$ECO/runs/pairs_hard_v4_holdout_clean" \
    --device cuda --out "$RUNS/eval_tc_v5_pairsv4.json"

echo "ALL_DONE"
