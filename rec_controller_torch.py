
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
from agent.ppo_rec import PPO, Memory
# from base_controller import BaseController
from env.base_env import BaseEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def one_hot(n, i):
    x = np.zeros(n)
    if i < n:
        x[i] = 1.
    return x

class NaiveController():
    def __init__(self, env: BaseEnv, max_episodes, lr, betas, gamma, clip_eps, n_steps, network_width, test_every, seed=None):
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
        self.n_steps = n_steps         # max timesteps in one episode
        self.n_latent_var = network_width           # number of variables in hidden layer
        self.update_timestep = 100      # update policy every n timesteps
        self.lr = lr
        self.betas = betas
        self.gamma = gamma                # discount factor
        self.K_epochs = 10                # update policy for K epochs
        self.eps_clip = clip_eps              # clip parameter for PPO
        self.random_seed = seed
        #############################################
        
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            # env.seed(random_seed)
        
        self.memorys = []
        self.ppos = []
        # print(self.env.observation_spaces[0])
        # print(self.env.observation_spaces[0].shape)
        # print(self.env.action_spaces[0])
        # print(self.env.action_spaces[0].n)

        for i in range(env.n_agents):
            curmemory = []
            curppo = []
            # for j in range(env.n_steps):
            memory = Memory()
            state_dim = self.env.observation_spaces[i].shape[0]
            action_dim = self.env.action_spaces[i].n
            ppo = PPO(state_dim, action_dim, self.n_latent_var, lr, betas[i], gamma, self.K_epochs, self.eps_clip, self.env.n_targets)
                # curmemory.append(memory)
                # curppo.append(ppo)
            self.memorys.append(memory)
            self.ppos.append(ppo)
        # print(lr,betas)

        self.env_prior = np.copy(self.env.prior)
        self.atk_prior = np.array([(1. / self.env.n_targets) for _ in range(self.env.n_targets)])
        
    
    def generate_prior(self):
        x = [0.] + sorted(np.random.rand(self.env.n_targets - 1).tolist()) + [1.]
        prior = np.zeros(self.env.n_targets)
        for i in range(self.env.n_targets):
            prior[i] = x[i + 1] - x[i]
        return prior

    def _get_atk_ob(self, target, belief, last_obs_atk) :
        ret = np.copy(last_obs_atk)
        # print(ret.shape)
        ret = ret[5:]
        ret = np.array([np.concatenate((belief, one_hot(self.env.n_targets, target), ret))])
        # print('ret shape:')
        # print(ret.shape)
        return ret    
    
        # training loop
    def train(self, num_round=None, step_each_round = 2000, policy_store_every=100, 
               sec_prob=False, save_every=None, save_path=None, load_state=None, load_path=None, store_results=False):
        self.step = 0
        env = self.env
        if num_round == None:
            num_round = self.max_episodes
        
        running_reward = np.zeros((self.env.n_agents))
        avg_length = 0
        timestep = 0
        

        while self.step < num_round:
            self.step += 1
            update_agent_num = int(self.step / self.update_timestep) % 2
            if update_agent_num == 0:
                self.env.set_prior(self.atk_prior)
            else:
                self.env.set_prior(self.env_prior)

            states, _, _ = self.env.reset()
            type_ob = np.zeros((1, self.env.n_targets))
            type_ob[0, self.env.atk_type] = 1.
            type_ob = torch.from_numpy(type_ob).float().to(device)
            # print('state shape:')
            # print(states[0].shape)
            rnn_historys = [torch.from_numpy(np.zeros((1, 1, self.n_latent_var))).float().to(device) for _ in range(self.env.n_agents)]
            
            # rnn_history = torch.from_numpy(rnn_history).float().to(device)
            
            current_len = 0

            while True:
                timestep += 1
                current_len += 1
                # curstep = substep
                
                actions = []

                for i in range(self.env.n_agents):
                    # action = self.ppos[i].act(curstep, states[i], self.memorys[i])
                    action, rnn_historys[i], _ = self.ppos[i].policy.act(states[i], rnn_historys[i], self.memorys[i])
                    actions.append(action)
                
                # atk_prob = [self.ppos[0].evaluate(t, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0])[3].cpu().detach().numpy() for tar in range(self.env.n_targets)]
                atk_prob = 0
                with torch.no_grad():
                    states, reward, done, _ = env.step(actions, atk_prob, False)
                
                # Saving reward and is_terminal:
                for i in range(self.env.n_agents):
                    self.memorys[i].rewards.append(reward[i])
                    self.memorys[i].is_terminals.append(done)
                    self.memorys[i].type_obs.append(type_ob[0])
                
                # update if its time
                if done and self.step % self.update_timestep == 0:
                    train_agent_n = int(self.step / self.update_timestep) % 2
                    # for i in range(self.env.n_agents):
                    v_loss = self.ppos[train_agent_n].update(self.memorys[train_agent_n])
                    for agent_i in range(self.env.n_agents):
                        self.memorys[agent_i].clear_memory()
                    print('timestep{} updated with q loss{}'.format(timestep, v_loss))
                    # timestep = 0
                
                running_reward += reward

                if done:
                    # for i in range(self.env.n_agents):
                        # self.ppos[i].update(self.memorys[i])
                        # self.memorys[i].clear_memory()
                    # timestep = 0
                    break

            avg_length += current_len
            current_len = 0
            
            
            # logging
            if self.step % self.log_interval == 0:

                avg_length = int(avg_length/self.log_interval)
                running_reward /= self.log_interval
                # running_reward = int((running_reward/self.log_interval))

                
                # print('Episode {} \t episode length: {} \t reward:'.format(self.step, avg_length))
                # print(running_reward)
                running_reward = np.zeros((self.env.n_agents))
                avg_length = 0