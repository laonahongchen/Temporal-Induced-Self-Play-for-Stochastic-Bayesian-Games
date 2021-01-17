#!/bin/bash

n_steps=5
now=$(date +"%Y-%m-%d-%T")
batch_size=1000
max_steps=1000000
# max_steps=1000
exp_name="models-r-$max_steps-bsize-$batch_size-time-$now"
# exp_name="models-r-200000-bsize-1000-time-2021-01-09-18:32:33"


for ((i=$n_steps-1; i >=0; i--))
do
    python run_tag_npa_parallel.py --num-round $i --n-steps $n_steps --max-process 10 --total-process 100 --v-batch-size 150000 --batch-size $batch_size --k-epochs 100 --max-steps $max_steps --exp-name $exp_name --learning-rate=1e-4
done
