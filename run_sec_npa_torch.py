# from env.matrix_env import MatrixEnv
# from env.tagging import TaggingEnv
from env.sec_belief import SecurityEnv
from npa_controller_torch import NaiveController
# import seaborn as sns
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from decimal import Decimal
from common.path_utils import *
import joblib
import argparse
import time


ppo_agent_cnt = 0
pac_agent_cnt = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run security game.")

    parser.add_argument('--agent', type=str, default="ppo")
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--n-steps', type=int, default=1)
    parser.add_argument('--n-belief', type=int, default=10)
    parser.add_argument('--steps-per-round', type=int, default=5)
    parser.add_argument('--prior', type=float, nargs='+', default=[0.5, 0.5])
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--minibatch', type=int, default=100)
    parser.add_argument('--test-every', type=int, default=5)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--load', action="store_true")
    parser.add_argument('--random-prior', action="store_true")
    parser.add_argument('--load-step', type=int)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--network-width', type=int, default=32)
    parser.add_argument('--network-depth', type=int, default=2)
    parser.add_argument('--sub-load-path', type=str)
    parser.add_argument('--timesteps-per-batch', type=int, default=8)
    parser.add_argument('--iterations-per-round', type=int, default=16)
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--other', type=str, default='')
    parser.add_argument('--seed', type=int, default=6657)

    return parser.parse_args()

if __name__ == "__main__":

    def argument_to_tuple(argument):
        if type(argument) == list and len(argument) == 1:
            return argument[0]
        elif type(argument) == str:
            return argument
        else:
            parameters = list(map(float, argument[1:]))
            return tuple(argument[:1] + parameters)

    args = parse_args()

    # seed = "benchmark"
    agent = args.agent
    n_steps = args.n_steps
    steps_per_round = args.steps_per_round
    prior = args.prior
    # prior = [0.2] * 5
    lr = learning_rate = args.learning_rate
    # schedule = ("wolf_adv", 20.0)
    test_every = args.test_every
    save_every = args.save_every
    load = args.load
    load_step = args.load_step
    max_steps = args.max_steps
    network_width = args.network_width
    network_depth = args.network_depth
    timesteps_per_batch = args.timesteps_per_batch
    iterations_per_round = args.iterations_per_round
    betas = [[0, 0.999], [0, 0.99]]
    gamma = 0.95
    max_episodes = args.episodes
    clip_eps = 0.2
    n_belief = args.n_belief

    priors = [[0.1 * i, 1 - 0.1 * i] for i in range(4, 7)]

    # other = "1000-test-steps-large-network"

    result_folder = "../result/"
    plot_folder = "../plots/"
    exp_name = args.exp_name or \
        "_".join(["deceive" + str(args.other),
                  "recurrent",
                  agent,
                  "game:{}-{}-{}".format(n_steps, steps_per_round, ":".join(map(str, prior)) if not args.random_prior else "random"),
                  "{:.0e}".format(Decimal(learning_rate)),
                  "test_every:{}".format(test_every),
                  "network:{}-{}".format(network_width, network_depth),
                  "train:{}*{}".format(timesteps_per_batch, iterations_per_round),
                  "start:{}".format(time.time())])

    exp_dir = os.path.join(result_folder, exp_name)
    plot_dir = os.path.join(plot_folder, exp_name)

    train = True

    res = {"episode": [], "current_assessments": [], "player": []}

    tot_res = []

    for i in range(1):

        env = SecurityEnv(n_slots=2,n_types=2,n_rounds=n_steps, prior=prior,zero_sum=True,seed=args.seed + i)

        for prior in priors:
            print('start with prior:')
            print(prior)

            env.set_prior(prior)

            # env.export_payoff("/home/footoredo/playground/REPEATED_GAME/EXPERIMENTS/PAYOFFSATTvsDEF/%dTarget/inputr-1.000000.csv" % n_slots)
            if train:
                controller = NaiveController(env, max_episodes, lr, betas, gamma, clip_eps, n_steps, network_width, test_every, n_belief, args.batch_size, args.minibatch, args.seed)
                controller.train(num_round=1, round_each_belief = 100000)

                
                print('train finish')

                strategies = controller.ppos[0], controller.ppos[1]
                tot_res.append(env.assess_strategies(strategies))
                print(tot_res)
    
    print(tot_res)

# 6657 5410 5748 9418 9527


# 21/sec_2: 2 3: 4.2156
# 21/sec: 2 5: 7.7078
# 14/npa_2: 2 8: 20.3159

'''
 5 5： 

npa:
PBNE: [tensor(6.8976, grad_fn=<SubBackward0>), tensor(15.9111, grad_fn=<SubBackward0>)] tensor(8.0222, grad_fn=<RsubBackward1>)                                                                                                                 BR: [tensor(22.4417, grad_fn=<AddBackward0>), tensor(31.7153, grad_fn=<AddBackward0>)] -6.942926804836428               Overall: [tensor(6.6375, grad_fn=<AddBackward0>), tensor(15.9111, grad_fn=<AddBackward0>)] tensor(8.0222, grad_fn=<RsubBackward1>)   
sec: 
PBNE: [tensor(9.7253, device='cuda:2', grad_fn=<SubBackward0>), tensor(11.2245, device='cuda:2', grad_fn=<SubBackward0>)] tensor(1.8543, device='cuda:2', grad_fn=<RsubBackward1>)                                                              BR: [tensor(30.9328, device='cuda:2', grad_fn=<AddBackward0>), tensor(33.0172, device='cuda:2', grad_fn=<AddBackward0>)] -19.62590601456302                                                                                                     Overall: [tensor(9.7253, device='cuda:2', grad_fn=<AddBackward0>), tensor(11.2245, device='cuda:2', grad_fn=<AddBackward0>)] tensor(1.8543, device='cuda:2', grad_fn=<RsubBackward1>)  


5 10:
21/sec_2: npa: 

10 10:

12/npa
'''