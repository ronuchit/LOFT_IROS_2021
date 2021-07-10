#!/bin/bash

START_SEED=1
NUM_SEEDS=1

ENV=cover  # cover or painting or blocks
COLLECT_DATA=0  # 1 or 0, whether to run data collection

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do

    python main.py  --env $ENV --start_seed $SEED --num_seeds 1 --collect_data $COLLECT_DATA

done
