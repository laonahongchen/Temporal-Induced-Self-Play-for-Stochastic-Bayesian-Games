# Backward_induction_RL

This is the code repo for the paper 'Temporal Induced Self-Play for Stochastic Bayesian Games' published in IJCAI21.

# Main dependencies: 

```
python>=3.8
torch
numpy
pandas
pickle
```


# Usage

To produce the result for TISP-PG in the tagging game, run `run_tag.sh`. This is a script that use a loop to train different rounds of the whole model with sequentially, to mitigate the garbish collection problem.

To produce the result for other method or our model in security game, use python commands with specific parameters, for example, to run backward induction method (BI) in security game, use the following command:

```shell
python run_sec_bi.py --n-steps 10 --learning-rate 1e-4 --max-steps 100000
```

To test the models in tagging game, use `run_tag_test.py` with the model name, for example, our models are test with the following command on our server:

```shell
python run_tag_test.py --exp-name models-r-800000-bsize-1000-time-2021-01-17-15:38:55 --n-steps 5
```
