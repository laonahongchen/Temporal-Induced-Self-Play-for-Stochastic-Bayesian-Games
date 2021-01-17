#!/bin/bash

n_steps=5
# exp_name="models-r-200000-bsize-1000-time-2021-01-07-18:20:15"
# exp_name="models-r-200000-bsize-1000-time-2021-01-08-11:05:47"
# exp_name="models-r-200000-bsize-1000-time-2021-01-06-19:03:13"
# exp_name="models-r-100000-bsize-1000-time-2021-01-08-14:40:23"
# exp_name="models-r-100000-bsize-1000-time-2021-01-08-16:44:35"
# exp_name="models-r-200000-bsize-1000-time-2021-01-08-17:03:54"
# exp_name="models-r-100000-bsize-1000-time-2021-01-09-15:47:12"
# exp_name="models-r-200000-bsize-1000-time-2021-01-09-18:32:33"
# exp_name="models-r-200000-bsize-1000-time-2021-01-11-16:23:39"
exp_name="models-r-500000-bsize-1000-time-2021-01-11-22:48:28"
now=$(date +"%Y-%m-%d-%T")
batch_size=1000
max_steps=200000
# max_steps=1000

python run_tag_test.py --n-steps $n_steps --batch-size $batch_size --exp-name $exp_name
