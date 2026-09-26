#!/usr/bin/env bash
#SBATCH --job-name=r2-spanpool
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/beegfs/a474r867/sciev/logs/r2sp_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/logs/r2sp_%j.err
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

# --- R2 spanpool: freeze + full-FT on two backbones ---
run -m sciev.train --pairs-train $TRAIN \
    --ckpt $RUNS/f0-span-esqueleto/checkpoint-g10000/model.pt \
    --config $BASE/harness/configs/f0-span-esqueleto.yaml \
    --out $RJ/runs/r2/esq_sp_freeze --steps 8000 --accum 8 --freeze \
    --r2-mode spanpool --device cuda
run -m sciev.train --pairs-train $TRAIN \
    --ckpt $RUNS/f0-span-esqueleto/checkpoint-g10000/model.pt \
    --config $BASE/harness/configs/f0-span-esqueleto.yaml \
    --out $RJ/runs/r2/esq_sp_ft --steps 8000 --accum 8 --lr 2e-5 \
    --r2-mode spanpool --device cuda
run -m sciev.train --pairs-train $TRAIN \
    --ckpt $RUNS/f2-spanes/checkpoint-g50000/model.pt \
    --config $BASE/harness/configs/f0-span-esqueleto.yaml \
    --out $RJ/runs/r2/f2_sp_freeze --steps 8000 --accum 8 --freeze \
    --r2-mode spanpool --device cuda
run -m sciev.train --pairs-train $TRAIN \
    --ckpt $RUNS/f2-spanes/checkpoint-g50000/model.pt \
    --config $BASE/harness/configs/f0-span-esqueleto.yaml \
    --out $RJ/runs/r2/f2_sp_ft --steps 8000 --accum 8 --lr 2e-5 \
    --r2-mode spanpool --device cuda

# --- eval each trained decision model on both batteries ---
for TAG in esq_sp_freeze esq_sp_ft f2_sp_freeze f2_sp_ft; do
  CKPT=$RJ/runs/r2/$TAG/decision.pt
  CFG=$BASE/harness/configs/f0-span-esqueleto.yaml
  for BAT in pairs_hard_v3_eval pairs_hard_v4_holdout_clean; do
    run -m sciev.eval --ckpt $CKPT --config $CFG \
        --pairs-dir $RUNS/$BAT --device cuda --r2-mode spanpool \
        --out $RJ/runs/r2/${TAG}__${BAT}.json
  done
done
echo "R2_SPANPOOL_COMPLETE"
