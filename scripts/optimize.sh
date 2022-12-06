#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate chemprobe

for FOLD in {0..4}
do
    python optimize.py \
    --study_path /srv/danger/scratch/wconnell/chemprobe/optuna/exp=$1 \
    --data_path ../data/preprocessed \
    --exp $1 \
    --fold $FOLD \
    --ntrials 5 \
    --prune \
    --batch_size 16384 \
    --gpus $2,
done