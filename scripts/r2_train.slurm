#!/usr/bin/env bash
#SBATCH --job-name=r2-marker
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/beegfs/a474r867/sciev/logs/r2_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/logs/r2_%j.err
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
RJ=/beegfs/a474r867/sciev
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
PY=/bb/bwvenv/bin/python
RUNS=$BASE/runs
TRAIN=$RUNS/pairs_hard_v3
mkdir -p $RJ/runs/r2
export PYTHONPATH=$RJ

run () { apptainer exec --nv --overlay "$OVL":ro "$SIF" "$PY" "$@"; }

# --- R2 marker-head training: freeze + full-FT on two backbones ---
run -m sciev.train --pairs-train $TRAIN \
    --ckpt $RUNS/f0-span-esqueleto/checkpoint-g10000/model.pt \
    --config $BASE/harness/configs/f0-span-esqueleto.yaml \
    --out $RJ/runs/r2/esq_freeze --steps 6000 --accum 8 --freeze --device cuda
run -m sciev.train --pairs-train $TRAIN \
    --ckpt $RUNS/f0-span-esqueleto/checkpoint-g10000/model.pt \
    --config $BASE/harness/configs/f0-span-esqueleto.yaml \
    --out $RJ/runs/r2/esq_ft --steps 6000 --accum 8 --lr 2e-5 --device cuda
run -m sciev.train --pairs-train $TRAIN \
    --ckpt $RUNS/f2-spanes/checkpoint-g50000/model.pt \
    --config $BASE/harness/configs/f0-span-esqueleto.yaml \
    --out $RJ/runs/r2/f2_freeze --steps 6000 --accum 8 --freeze --device cuda
run -m sciev.train --pairs-train $TRAIN \
    --ckpt $RUNS/f2-spanes/checkpoint-g50000/model.pt \
    --config $BASE/harness/configs/f0-span-esqueleto.yaml \
    --out $RJ/runs/r2/f2_ft --steps 6000 --accum 8 --lr 2e-5 --device cuda

# --- eval each trained decision model on both batteries ---
for TAG in esq_freeze esq_ft f2_freeze f2_ft; do
  CKPT=$RJ/runs/r2/$TAG/decision.pt
  CFG=$BASE/harness/configs/f0-span-esqueleto.yaml
  for BAT in pairs_hard_v3_eval pairs_hard_v4_holdout_clean; do
    run -m sciev.eval --ckpt $CKPT --config $CFG \
        --pairs-dir $RUNS/$BAT --device cuda \
        --out $RJ/runs/r2/${TAG}__${BAT}.json
  done
done
echo "R2_COMPLETE"
