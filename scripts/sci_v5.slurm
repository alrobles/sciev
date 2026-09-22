#!/bin/bash
#SBATCH --job-name=sci_v5
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.err
#
# Spanpool-FT on the *scientific* decision battery, init from v5-1b.
# Submit with:  sbatch --dependency=afterany:<v5-1b jobids> scripts/sci_v5.slurm

set -euo pipefail
RJ=/beegfs/a474r867/reverse-jev
ECO=/beegfs/a474r867/ecoreasoner
BAT=$RJ/data/sci_battery
RUNS=$RJ/runs/sci
SIF=$ECO/pytorch-cuda.sif
OVL=$ECO/blackwell-torch.overlay
PY=/bb/bwvenv/bin/python
CFG=$ECO/harness/configs/eval-moe-v5.yaml
mkdir -p "$RUNS"
export HF_HOME=/beegfs/a474r867/hf-cache
cd "$RJ"

CKPT=""
for d in $(ls -d $ECO/runs/v5-1b/checkpoint-g*/ 2>/dev/null | sort -t g -k2 -n -r); do
    f="${d}model.pt"
    if [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 4000000000 ]; then
        CKPT="$f"; break
    fi
done
[ -n "$CKPT" ] || { echo "no complete v5-1b checkpoint found"; exit 1; }
echo "=== v5-1b checkpoint: $CKPT ==="

cat "$BAT"/sci_choice_train.jsonl \
    "$BAT"/sci_noul_train.jsonl \
    "$BAT"/sci_score_train.jsonl > "$BAT"/sci_all_train.jsonl
cat "$BAT"/sci_choice_dev.jsonl \
    "$BAT"/sci_noul_dev.jsonl \
    "$BAT"/sci_score_dev.jsonl > "$BAT"/sci_all_dev.jsonl

run() { apptainer exec --nv --overlay "$OVL":ro "$SIF" "$PY" "$@"; }

echo "=== TRAIN sci spanpool ft (v5-1b init) ==="
run -m reverse_jev.train \
    --decisions-train "$BAT/sci_all_train.jsonl" \
    --ckpt "$CKPT" --config "$CFG" \
    --r2-mode spanpool --steps 8000 --accum 1 \
    --lr 2e-5 --head-lr 3e-4 --warmup 200 --device cuda \
    --out "$RUNS/sci_v5_sp"

for KIND in choice noul score; do
    echo "=== EVAL $KIND (sci_v5_sp) ==="
    run -m reverse_jev.eval \
        --ckpt "$RUNS/sci_v5_sp/decision.pt" --config "$CFG" \
        --r2-mode spanpool \
        --r2-temp-fit-decisions "$BAT/sci_all_dev.jsonl" \
        --decisions-eval "$BAT/sci_${KIND}_eval.jsonl" \
        --device cuda --out "$RUNS/eval_sci_v5_${KIND}.json"
done
echo "ALL_DONE"
