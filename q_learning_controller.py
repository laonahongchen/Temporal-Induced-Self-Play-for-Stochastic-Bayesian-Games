import argparse
import pickle
from collections import namedtuple

import os
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# from agent.ppo_torch import PPO, Memory
# from agent.bi_torch import BackInductionAgent, Memory
# from agent.npa_torch import NPAAgent, Memory
from agent.table_q import TableQAgent, AtkQAgent
# from base_controller import BaseController
from env.base_env import BaseEnv

class NaiveController():
    def __init__(self, env: BaseEnv, max_episodes, lr, gamma, test_every, seed=None):
    ############## Hyperparameters ##############
        # env_name = "LunarLander-v2"
        # creating environment
        # env = gym.make(env_name)
        self.env = env
        # state_dim = env.observation_space.shape[0]
        # action_dim = 4
        self.render = False
        # solved_reward = 230         # stop training if avg_reward > solved_reward
        self.log_interval = test_every           # print avg reward in the interval
        self.max_episodes = max_episodes        # max training episodes
        self.lr = lr
        # self.betas = betas
        self.gamma = gamma                # discount factor
        self.random_seed = seed
        #############################################
        
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            # env.seed(random_seed)
        
        self.agents = []
        # self.atk_types = []
       # for i in range(env.n_targets):
        state_dim = self.env.observation_spaces[0].shape[0]
        action_dim = self.env.action_spaces[0].n
        ppo = AtkQAgent(self.env.n_types, state_dim, action_dim, lr, gamma)
            # self.atk_types.append(ppo)

        self.agents.append(ppo)
        # for i in range(env.n_agents):
        state_dim = self.env.observation_spaces[1].shape[0]
        action_dim = self.env.action_spaces[1].n
        ppo = TableQAgent(state_dim, action_dim, lr, gamma)
        self.agents.append(ppo)
        # print(lr,betas)
    
        # training loop
    def train(self, num_round = None):
        
        for num_episode in range(num_round):
            episodic_reward = [0 for _ in range(self.env.n_agents)]
            is_terminated = False
            s0, _, _ = self.env.reset()
            # self.agents[0] = self.atk_types[0]

            while not is_terminated:
                actions = []
                for i in range(self.env.n_agents):
                    action = self.agents[i].act(s0[i])
                    actions.append(action)
                s1, reward, is_terminated, _ = self.env.step(actions, 0)
                episodic_reward += reward
                for i in range(self.env.n_agents):
                    self.agents[i].update(actions[i], s0[i], s1[i], reward[i], is_terminated)
                s0 = s1
            
            if num_episode % self.log_interval == 0:
                print('{} episodes trained'.format(num_episode))
                strategy = self.agents[0], self.agents[1]
                self.env.assess_strategies(strategy)
                # print("Episode: {}, Score: {}".format(num_episode, episodic_reward))
            # env.reset()