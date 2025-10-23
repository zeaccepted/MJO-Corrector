#!/bin/bash
t=`date '+%Y%m%d_%H%M%S'`;
echo $t;

export CUDA_VISIBLE_DEVICES=0
chmod +x train_model1.py


log_file="train_model1.out"


python -u train_model1.py > $log_file 2>&1 &

echo "Training started, log file: $log_file"