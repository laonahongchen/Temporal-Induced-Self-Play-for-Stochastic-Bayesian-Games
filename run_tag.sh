#!/bin/bash

n_steps=5
now=$(date +"%Y-%m-%d-%T")
batch_size=1000
max_steps=100000

for ((i=$n_steps-1; i >=0; i--))
do
    python run_tag_npa_parallel.py --num-round $i --n-steps $n_steps --max-process 3 --batch-size $batch_size --k-epochs 40 --max-steps $max_steps --exp-name "models-r-$max_steps-bsize-$batch_size-time-$now"
done