#!/bin/bash
#SBATCH --job-name=e0-holdout-cpu
#SBATCH --partition=sixhour
#SBATCH --time=5:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/beegfs/a474r867/reverse-jev/logs/e0hold_%j.out
#SBATCH --error=/beegfs/a474r867/reverse-jev/logs/e0hold_%j.err

BASE=/beegfs/a474r867/ecoreasoner
RJ=/beegfs/a474r867/reverse-jev
IMG=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
PY=/bb/bwvenv/bin/python

mkdir -p $RJ/runs/e0 $RJ/logs
cd $RJ

run_eval () {
  CKPT=$1; NAME=$2; PAIRS=$3; BAT=$4
  echo "--- $NAME vs $BAT ---"
  apptainer exec --bind /beegfs:/beegfs --overlay $OVL $IMG \
    $PY -m reverse_jev.eval \
      --ckpt $CKPT --pairs $PAIRS --device cpu \
      --out $RJ/runs/e0/${NAME}__${BAT}__cpu.json
}

PAIRS_V4=$BASE/runs/pairs_hard_v4_holdout_clean
run_eval $BASE/runs/f0-span-esqueleto/checkpoint-g10000/model.pt \
         f0-span-esqueleto $PAIRS_V4 pairs_hard_v4_holdout_clean
run_eval $BASE/runs/f0-span-v2-weight-tying/checkpoint-g10000/model.pt \
         f0-span-v2-weight-tying $PAIRS_V4 pairs_hard_v4_holdout_clean
run_eval $BASE/runs/f2-spanes/checkpoint-g50000/model.pt \
         f2-spanes-50k $PAIRS_V4 pairs_hard_v4_holdout_clean

echo "E0_HOLDOUT_COMPLETE"
