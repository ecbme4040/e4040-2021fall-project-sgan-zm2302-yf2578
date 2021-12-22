#!/bin/bash

experiment_dir="experiments/ngsim/ngsim_v_dt=1"
data_dir="data/ngsim/ngsim_v_dt=1"
mode="train_and_test"

python3 main_lstm.py --experiment_dir $experiment_dir --data_dir $data_dir --mode $mode --force_overwrite
python3 viz.py --experiment_dir $experiment_dir --mode "debug" --sudoku --interval --metrics_statistic --force_overwrite