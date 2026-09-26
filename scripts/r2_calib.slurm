#!/usr/bin/env bash
#SBATCH --job-name=r2-calib
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/beegfs/a474r867/sciev/logs/r2cal_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/logs/r2cal_%j.err
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
RJ=/beegfs/a474r867/sciev
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
PY=/bb/bwvenv/bin/python
RUNS=$BASE/runs
TRAIN=$RUNS/pairs_hard_v3
DEV=$RUNS/pairs_hard_v3_eval
TEST=$RUNS/pairs_hard_v4_holdout_clean
CFG=$BASE/harness/configs/f0-span-esqueleto.yaml
mkdir -p $RJ/runs/r2
export PYTHONPATH=$RJ

run () { apptainer exec --nv --overlay "$OVL":ro "$SIF" "$PY" "$@"; }

# --- RCDL-lite: f2-50k + spanpool + FT + REINFORCE scoring rule ---
run -m sciev.train --pairs-train $TRAIN \
    --ckpt $RUNS/f2-spanes/checkpoint-g50000/model.pt \
    --config $CFG \
    --out $RJ/runs/r2/f2_sp_ft_rl --steps 8000 --accum 8 --lr 2e-5 \
    --r2-mode spanpool --rl 0.5 --rl_samples 8 --rl_noise 0.4 --device cuda

# --- calibration: T fitted on DEV, applied to TEST (holdout stays clean) ---
for TAG in f2_sp_ft f2_sp_ft_rl; do
  CKPT=$RJ/runs/r2/$TAG/decision.pt
  for BAT in pairs_hard_v3_eval pairs_hard_v4_holdout_clean; do
    run -m sciev.eval --ckpt $CKPT --config $CFG \
        --pairs-dir $RUNS/$BAT --device cuda --r2-mode spanpool \
        --out $RJ/runs/r2/${TAG}__${BAT}__raw.json
    run -m sciev.eval --ckpt $CKPT --config $CFG \
        --pairs-dir $RUNS/$BAT --device cuda --r2-mode spanpool \
        --r2-temp-fit $DEV \
        --out $RJ/runs/r2/${TAG}__${BAT}__temp.json
  done
done
echo "R2_CALIB_COMPLETE"
