#!/usr/bin/env bash
#SBATCH --job-name=e0-holdout-cpu
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/sciev/logs/e0hold_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/logs/e0hold_%j.err
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
RJ=/beegfs/a474r867/sciev
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
PY=/bb/bwvenv/bin/python
RUNS=$BASE/runs
OUT=$RJ/runs/e0
mkdir -p "$OUT"
export PYTHONPATH=$RJ
export OMP_NUM_THREADS=16

SPECS="f0-span-esqueleto|/beegfs/a474r867/ecoreasoner/runs/f0-span-esqueleto/checkpoint-g10000/model.pt|/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-esqueleto.yaml
f0-span-v2-weight-tying|/beegfs/a474r867/ecoreasoner/runs/f0-span-v2-weight-tying/checkpoint-g10000/model.pt|/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-v2-weight-tying.yaml
f2-spanes-50k|/beegfs/a474r867/ecoreasoner/runs/f2-spanes/checkpoint-g50000/model.pt|/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-esqueleto.yaml"

PAIRS=pairs_hard_v4_holdout_clean
echo "=== BATTERY $PAIRS (CPU) ==="
echo "$SPECS" | while IFS='|' read -r tag ckpt cfg; do
    [ -z "$tag" ] && continue
    echo "--- $tag vs $PAIRS ---"
    apptainer exec --overlay "$OVL":ro "$SIF" \
        "$PY" -m sciev.eval \
        --ckpt "$ckpt" --config "$cfg" \
        --pairs-dir "$RUNS/$PAIRS" \
        --device cpu \
        --out "$OUT/${tag}__${PAIRS}__cpu.json"
done
echo "E0_HOLDOUT_COMPLETE"
