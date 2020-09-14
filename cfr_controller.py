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
from agent.cfr import NPAAgent, AtkNPAAgent
from env.base_env import BaseEnv
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def one_hot(n, i):
    x = np.zeros(n)
    if i < n:
        x[i] = 1.
    return x

class NaiveController():
    def __init__(self, env: BaseEnv, max_episodes, lr, betas, gamma, clip_eps, n_steps, network_width, test_every, n_belief, n_states, batch_size, minibatch, k_epochs, v_epochs, seed=None):
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
        self.update_timestep = batch_size      # update policy every n timesteps
        self.minibatch = minibatch
        self.lr = lr
        self.betas = betas
        self.gamma = gamma                # discount factor
        self.K_epochs = k_epochs                # update policy for K epochs
        self.v_epochs = v_epochs
        self.eps_clip = clip_eps              # clip parameter for PPO
        self.random_seed = seed
        self.n_belief = n_belief
        self.n_state = n_states
        #############################################
        
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            # env.seed(random_seed)
        
        self.memorys = []
        self.agents = []
        # print(self.env.observation_spaces[0])
        # print(self.env.observation_spaces[0].shape)
        # print(self.env.action_spaces[0])
        # print(self.env.action_spaces[0].n)

        # self.beliefs = [[self.generate_prior() for i in range(self.n_belief)] for j in range(self.env.n_steps)]
        # self.beliefs = [self.generate_belief(self.n_belief) for j in range(self.env.n_steps)]
        self.beliefs = [[np.array([(1. / self.n_belief) * i, 1 - (1. / self.n_belief) * i]) for i in range(self.n_belief)] for j in range(self.env.n_steps)]
        # print('beliefs:')
        # print(self.beliefs)
        atk_belief_n = []
        for i in range(self.env.n_targets):
            new_belief = np.zeros(self.env.n_targets)
            new_belief[i] = 1
            atk_belief_n.append(np.copy(new_belief))
        self.atk_belief = [atk_belief_n for i in range(self.env.n_steps)]
        # print('atk belief:')
        # print(self.atk_belief)
        self.beliefs_n = [self.atk_belief, self.beliefs]
        self.n_belief_n = [self.env.n_targets, self.n_belief]
        self.env_prior = np.copy(self.env.prior)
        self.atk_prior = np.array([(1. / self.env.n_targets) for _ in range(self.env.n_targets)])
        self.action_dims = []

        for i in range(env.n_agents):
            # curmemory = []
            curppo = []
            # memory = Memory()
            state_dim = self.env.observation_spaces[i].shape[0]
            action_dim = self.env.action_spaces[i].n
            self.action_dims.append(action_dim)
            # ppo = SampleACAgent(self.env.n_steps, self.n_belief_n[i], self.beliefs_n[i], state_dim, action_dim, self.n_latent_var, lr, betas[i], gamma, self.K_epochs, self.eps_clip, self.minibatch, self.env.n_targets)
            if i != 0:
                ppo = NPAAgent(self.env.n_steps, self.n_belief, self.beliefs, state_dim, action_dim, self.n_latent_var, lr, betas[i], gamma, self.v_epochs)
            else:
                ppo = AtkNPAAgent(self.env.n_types, self.env.n_steps, self.n_belief, self.beliefs, state_dim, action_dim, self.n_latent_var, lr, betas[i], gamma, self.v_epochs)
            # self.memorys.append(memory)
            self.agents.append(ppo)
    
    def _get_atk_ob(self, target, belief, last_obs_atk) :
        ret = np.copy(last_obs_atk)
        ret = ret[1 + self.env.n_targets * 2:]
        ret = np.array([np.concatenate((belief, ret, one_hot(self.env.n_targets, target)))])
        return ret    

    def train(self, num_round=None, round_each_belief = 1000, policy_store_every=100, 
               sec_prob=False, save_every=None, save_path=None, load_state=None, load_path=None, store_results=False):
        self.step = 0
        env = self.env
        if num_round == None:
            num_round = self.max_episodes
        
        running_reward = np.zeros((self.env.n_targets, self.env.n_agents))
        avg_length = np.zeros((self.env.n_targets))
        timestep = 0
        epi_cnt = 0
        epi_type_cnt = np.zeros((self.env.n_targets))
        done_cnt = 0

        while self.step < num_round:
            self.step += 1
            for substep in range(env.n_steps - 1, -1, -1):
                print('start training substep {}.'.format(substep))

                for b in range(self.n_belief):
                    atk_obs = [[] for _ in range(self.env.n_types)]
                    def_obs = []
                    def_value_preds = []
                    atk_value_preds = [[] for _ in range(self.env.n_types)]
                    atk_p_res = [[] for _ in range(self.env.n_types)]
                    def_p_res = []
                    for i_state in range(self.n_state):
                        update_agent_num = 0
                        oppo_agent_num = 1
                        done_cnt = 0
                        train_state, _, _ = self.env.sub_reset(substep, self.beliefs[substep][b])
                        state_info, belief = self.env.get_cur_state()
                        def_obs.append(train_state[1][1 + self.env.n_targets:])

                        print('{}: start training in round {}, belief {}, state {}'.format(datetime.now(), substep, b, i_state))
                        atk_r = [-1000 for _ in range(self.env.n_types)]
                        def_r = -1000
                    
                        atk_s = []
                        def_s = self.agents[1].act(substep, train_state[1], b, True)[1].detach()

                        # print('def s:')
                        # print(def_s)

                        for type_i in range(self.env.n_types):
                            train_state, _, _ = self.env.reset_to_state_with_type(state_info, belief, type_i)
                            atk_s.append(self.agents[0].act(substep, train_state[0], type_i, b, True)[1].detach())

                        for i_episode in range(round_each_belief):
                            def_vs = [0 for _ in range(self.action_dims[1])]
                            atk_vs = [[0 for __ in range(self.action_dims[0])] for _ in range(self.env.n_types)]

                            for atk_action in range(self.action_dims[0]):
                                atk_prob = [self.agents[0].evaluate(substep, self._get_atk_ob(tar, self.env.belief, train_state[0]), atk_action, tar, b, True)[0][0].cpu().detach().numpy() for tar in range(self.env.n_targets)]
                                atk_tmpvs = [0 for _ in range(self.env.n_types)]
                                for def_action in range(self.action_dims[1]):
                                    def_sumv = 0
                                    atk_sumv = [0 for _ in range(self.env.n_types)]
                                    type_cnts = [0 for _ in range(self.env.n_types)]
                                    actions = [atk_action, def_action]
                                    
                                    for type_i in range(self.env.n_types):
                                        # print('cur type: {}'.format(type_i))
                                        for epoch_i in range(self.K_epochs):
                                            self.env.reset_to_state_with_type(state_info, belief, type_i)
                                            next_states, reward, done, _ = self.env.step(actions, atk_prob)
                                            tmpv = reward
                                            if not done:
                                                tmpv[0] += self.gamma * self.agents[0].evaluate(substep + 1, next_states[0][1:], 0, type_i, b,  False)[1].detach()
                                                tmpv[1] += self.gamma * self.agents[1].evaluate(substep + 1, next_states[1][1:], 0, b, False)[1].detach()
                                            atk_sumv[self.env.atk_type] += tmpv[0] * def_s[def_action] / self.K_epochs
                                            def_sumv += tmpv[1] * self.beliefs[substep][b][type_i] * atk_s[type_i][atk_action]/ self.K_epochs 
                                    def_vs[def_action] += def_sumv

                                    for type_i in range(self.env.n_types):
                                        atk_tmpvs[type_i] += atk_sumv[type_i]
                                for type_i in range(self.env.n_types):
                                    atk_vs[type_i][atk_action] = atk_tmpvs[type_i]# .append(atk_tmpvs[type_i])
                            # vs = np.array(vs)
                            def_vs = np.array(def_vs)
                            # def_vs -= def_r
                            atk_vs = np.array(atk_vs)
                            def_vs_t = def_vs - def_r
                            atk_vs_t = np.copy(atk_vs)
                            for type_i in range(self.env.n_types):
                                atk_vs_t[type_i] -= atk_r[type_i]
                            # atk_s = np.zeros(self.env.n_types)

                            self.agents[1].update(substep, train_state[1], b, def_vs_t)
                            self.agents[0].update(substep, train_state[0], b, atk_vs_t)

                            atk_s = []
                            def_s = self.agents[1].act(substep, train_state[1], b, True)[1].detach()
                            # cur_res = def_s * def_vs
                            def_r = torch.sum(def_s * torch.Tensor(def_vs)).item()
                            # print('def_r:')
                            # print(def_r)

                            for type_i in range(self.env.n_types):
                                train_state, _, _ = self.env.reset_to_state_with_type(state_info, belief, type_i)
                                atk_s.append(self.agents[0].act(substep, train_state[0], type_i, b, True)[1].detach())
                                # cur_res = atk_vs * atk_s[type_i]
                                atk_r[type_i] = torch.sum(torch.Tensor(atk_vs) * atk_s[type_i]).item()
                                # print(cur_res)

                            # print('cur r:')
                            # print(def_r, atk_r)
                            
                        
                        print(datetime.now(), ': pol updated')

                        # calculate the value function
                        def_value = 0
                        def_strategy = self.agents[1].act(substep, train_state[1], b)[1].detach()

                        for type_i in range(self.env.n_types):
                            ev_atk = 0
                            ev_def = 0
                            train_state,  _, _ = self.env.reset_to_state_with_type(state_info, belief, type_i)
                            atk_strategy = self.agents[0].act(substep, train_state[0], type_i, b)[1].detach()
                            atk_obs[type_i].append(train_state[0][1 + self.env.n_targets:])
                            
                            for atk_action in range(self.action_dims[0]):
                                # need to get atk_prob
                                # TODO update return and API
                                atk_prob = [self.agents[0].evaluate(substep, self._get_atk_ob(tar, self.env.belief, train_state[0]), atk_action, tar, b, in_training=True)[0].cpu().detach().numpy() for tar in range(self.env.n_targets)]
                                tmpvs = []
                                for def_action in range(self.action_dims[1]):
                                    sumv = None
                                    actions = [atk_action, def_action]
                                    # reward = []
                                    for epoch_i in range(self.K_epochs):
                                        self.env.reset_to_state_with_type(state_info, belief, type_i)
                                        # tmpv = [0, 0]
                                        next_states, reward, done, _ = self.env.step(actions, atk_prob)
                                        tmpv = reward
                                        if not done:
                                            tmpv[0] += self.gamma * self.agents[0].evaluate(substep + 1, next_states[0][1:], 0, type_i, b)[1].detach()
                                            tmpv[1] += self.gamma * self.agents[1].evaluate(substep + 1, next_states[1][1:], 0, b)[1].detach()
                                        if epoch_i > 0:
                                            sumv += tmpv
                                        else:
                                            sumv = np.array(tmpv)
                                    # sumv /= self.K_epochs
                                    ev_atk += sumv[0] * atk_strategy[atk_action] * def_strategy[def_action] / self.K_epochs
                                    ev_def += sumv[1] * atk_strategy[atk_action] * def_strategy[def_action] / self.K_epochs
                            # print('def value:')
                            # print(ev_def, self.beliefs[substep][b][type_i], def_value)
                            def_value += ev_def * self.beliefs[substep][b][type_i]
                            # atk_values.append(ev_atk)
                            atk_value_preds[type_i].append(ev_atk.detach())
                            
                            atk_p_res[type_i].append(self.agents[0].act(substep, train_state[0], type_i, b)[1].detach())
                        def_value_preds.append(def_value.detach())
                        def_p_res.append(def_strategy.detach())


                    self.agents[0].value_supervise(substep, b, atk_obs, atk_value_preds)
                    self.agents[0].policy_supervise(substep, b, atk_obs, atk_p_res)
                    self.agents[1].value_supervise(substep, b, def_obs, def_value_preds)
                    self.agents[1].policy_supervise(substep, b, def_obs, def_p_res)
                
    def assess_strategy(self, episodes_test=100):
        atk_rews = [0 for _ in range(self.env.n_types)]
        def_rews = [0 for _ in range(self.env.n_types)]
        type_cnt = [0 for _ in range(self.env.n_types)]
        for i in range(episodes_test):
            cur_round = 0
            states, _, _ = self.env.reset()
            cur_type = self.env.atk_type
            type_cnt[cur_type] += 1
            done = False
            atk_rew = 0
            def_rew = 0
            while not done:
                atk_action, _ = self.agents[0].act(cur_round, states[0], cur_type)
                def_action, _ = self.agents[1].act(cur_round, states[1])
                atk_prob = [self.agents[0].evaluate(substep, self._get_atk_ob(tar, self.env.belief, state[0]), atk_action, self.env.atk_type)[0][0].cpu().detach().numpy() for tar in range(self.env.n_targets)]
                actions = [atk_action, def_action]
                states, rew, done, _ = self.step(actions, atk_prob)
                atk_rew += rew[0]
                def_rew += rew[1]
            atk_rews += atk_rew
            def_rews += def_rew

        for type_i in range(self.env.n_types):
            if type_cnt[type_i] != 0:
                print('type {}: sampled {} times, atk_avg_rew: {}, def_avg_rew: {}'.format(type_i, type_cnt[type_i], atk_rews[type_i] / type_cnt[type_i], def_rews[type_i] / type_cnt[type_i]))