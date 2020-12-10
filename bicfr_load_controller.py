
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
from agent.ppo_torch import PPO as test_PPO
from agent.ppo_torch import Memory as test_Memory
from agent.ppo_rec import PPO, Memory, AtkNPAAgent
# from base_controller import BaseController
from env.base_env import BaseEnv
from tagging_cfr_executor import CFROppActor, CFRProActor

from datetime import datetime

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def one_hot(n, i):
    x = np.zeros(n)
    if i < n:
        x[i] = 1.
    return x

def save_model(model, f_path):
    with open(f_path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    print('model saved to {}'.format(f_path))

def load_model(f_path):
    with open(f_path, 'rb') as f:
        model = pickle.load(f)
    return model

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
        self.update_timestep = 500      # update policy every n timesteps
        self.lr = lr
        self.betas = betas
        self.gamma = gamma                # discount factor
        self.K_epochs = 1                # update policy for K epochs
        self.eps_clip = clip_eps              # clip parameter for PPO
        self.random_seed = seed
        #############################################
        
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            # env.seed(random_seed)
        
        self.ppos = []
        self.ppos.append(CFROppActor())
        
        self.memorys = []
        self.action_transformer = [3, 2, 0, 1, 4]
        # self.ppos = []
        # print(self.env.observation_spaces[0])
        # print(self.env.observation_spaces[0].shape)
        # print(self.env.observation_spaces[1].shape)
        # print(self.env.action_spaces[0])
        # print(self.env.action_spaces[0].n)

        # print(lr,betas)

        self.atk_memorys = [Memory() for _ in range(self.env.n_types)]
        self.def_memory = Memory()

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
        ret = ret[1 + self.env.n_targets: -self.env.n_targets]
        ret = np.array([np.concatenate((belief, ret, one_hot(self.env.n_targets, target)))])
        # print('ret shape:')
        # print(ret.shape)
        return ret
    
    def calculate_exploitability(self, single_train_round=100000, episodes_test=100, exp_name=None):
        # if exp_name != None:
        #     self.load_models(exp_name)
        self.step = 0
        env = self.env
        # if num_round == None:
            # num_round = self.max_episodes
        
        running_reward = np.zeros((self.env.n_agents))
        avg_length = 0
        timestep = 0
        state_dim = self.env.observation_spaces[1].shape[0]
        action_dim = self.env.action_spaces[1].n
        train_agent = test_PPO(state_dim + self.env.n_targets, action_dim, self.n_latent_var, self.lr, self.betas[1], self.gamma, self.K_epochs, self.eps_clip, self.env.n_targets)
        train_memory = test_Memory()

        # update_agent_num = 0

        while self.step < single_train_round:
            self.step += 1
            # update_agent_num = int(self.step / self.update_timestep) % 2
            # if update_agent_num == 0:
                # self.env.set_prior(self.atk_prior)
            # else:
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
                pre_rnns = []
                pre_rnns.append(rnn_historys[0])

                # for i in range(self.env.n_agents):
                #     # pre_rnns.append(np.copy(rnn_historys[i]))
                #     # pre_rnns.append(rnn_historys[i])
                #     # action = self.ppos[i].act(curstep, states[i], self.memorys[i])
                #     if i == 0:
                #         action, _ = self.ppos[i].act(current_len, states[i], self.atk_memorys[self.env.atk_type], self.env.atk_type)
                #     else:
                #         action, _ = self.ppos[i].act(current_len, states[i], self.def_memory)
                #     actions.append(action)
                # atk_action, rnn_historys[0], atk_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, rnn_historys[0], type_n=self.env.atk_type)
                atk_action = self.ppos[0].act(self.n_steps + 1 - current_len, self.env.atk_type, [states[0][1]], states[0][3], states[0][4], states[0][5], states[0][6])
                # print(atk_action)
                atk_action_n = self.action_transformer[atk_action]
                actions.append(atk_action_n)
                action, _ = train_agent.act(current_len, states[1], train_memory)
                actions.append(action)
                
                # v = []
                # if current_len > 1:
                    # for i in range(self.env.n_agents):
                    # print('state:')
                    # print(states[0][1:], np.array([states[0][1:]]))
                # v.append(self.ppos[0].evaluate(torch.stack([states[0]]), actions[0], type_ob, self.env.atk_type, in_training=True)[1])
                v = train_agent.evaluate(torch.stack([states[1]]), actions[1], type_ob, in_training=True)[1]
                
                # atk_prob = torch.stack([self.ppos[0].evaluate(self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], np.array([one_hot(self.env.n_targets, tar)]), tar, in_training=False)[1].detach() for tar in range(self.env.n_targets)])
                # atk_prob = torch.stack([self.ppos[0].evaluate(self._get_atk_ob(tar, self.env.belief, states[0]), pre_rnns[0], actions[0], one_hot(self.env.n_targets, tar), tar, in_training=True)[1].detach() for tar in range(self.env.n_targets)])
                # atk_prob = 0
                # with torch.no_grad():
                atk_prob = torch.Tensor([self.ppos[0].strategy(self.n_steps + 1 - current_len, tar, [states[0][1]], states[0][3], states[0][4], states[0][5], states[0][6])[atk_action] for tar in range(self.env.n_targets)])
                # print(atk_prob)
                states, reward, done, _ = env.step(actions, atk_prob)
                
                # Saving reward and is_terminal:
                # for i in range(self.env.n_agents):
                #     self.memorys[i].rewards.append(reward[i])
                #     self.memorys[i].is_terminals.append(done)
                #     self.memorys[i].type_obs.append(type_ob[0])
                
                train_memory.rewards.append(reward[1])
                train_memory.is_terminals.append(done)
                train_memory.type_obs.append(type_ob[0])
                if current_len > 1:
                    train_memory.next_vs.append(v)
                    # self.atk_memorys[self.env.atk_type].next_vs.append(v[0])
                if done:
                    train_memory.next_vs.append(torch.zeros_like(v))
                    # self.atk_memorys[self.env.atk_type].next_vs.append(torch.zeros_like(v[0]))

                # update if its time
                if done and self.step % self.update_timestep == 0:
                    # train_agent_n = int(self.step / self.update_timestep) % 2
                    # for i in range(self.env.n_agents):
                    # v_loss = self.ppos[train_agent_n].update(self.memorys[train_agent_n])
                    v_loss = 0
                    train_agent.update(train_memory)
                    # for agent_i in range(self.env.n_agents):
                        # self.memorys[agent_i].clear_memory()
                    # for type_i in range(self.env.n_types):
                        # self.atk_memorys[type_i].clear_memory()
                    train_memory.clear_memory()
                    # update_agent_num = (update_agent_num + 1) % 2
                    # if timestep % 
                    print('timestep{} updated with q loss{}'.format(self.step, v_loss))
                    # timestep = 0
                
                running_reward += reward

                if done:
                    # for i in range(self.env.n_agents):
                        # self.ppos[i].update(self.memorys[i])
                        # self.memorys[i].clear_memory()
                    # timestep = 0
                    break
        
        atk_rews = [0 for _ in range(self.env.n_types)]
        def_rews = [0 for _ in range(self.env.n_types)]
        type_cnt = [0 for _ in range(self.env.n_types)]
        for i in range(episodes_test):
            cur_round = 0
            states, _, _ = self.env.reset()
            print('new episode:')
            # states, _, _ = self.env.sub_reset(0, self.env.prior)
            cur_type = self.env.atk_type
            type_cnt[cur_type] += 1
            done = False
            atk_rew = 0
            def_rew = 0
            rnn_historys = [torch.from_numpy(np.zeros((1, 1, self.n_latent_var))).float().to(device) for _ in range(self.env.n_agents)]
            while not done:
                pre_rnns = []
                pre_rnns.append(rnn_historys[0])
                pre_rnns.append(rnn_historys[1])

                # atk_action, rnn_historys[0], atk_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, rnn_historys[0], type_n=cur_type)
                atk_action = self.ppos[0].act(self.n_steps + 1 - current_len, self.env.atk_type, [states[0][1]], states[0][3], states[0][4], states[0][5], states[0][6])
                atk_action_n = self.action_transformer[atk_action]
                def_action, def_strategy = train_agent.act(cur_round, states[1], train_memory)
                # atk_prob = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), atk_action, tar, np.array([one_hot(self.env.n_targets, tar)]))[1].detach() for tar in range(self.env.n_targets)])
                actions = [atk_action, def_action]
                atk_prob = torch.Tensor([self.ppos[0].strategy(self.n_steps + 1 - current_len, tar, [states[0][1]], states[0][3], states[0][4], states[0][5], states[0][6])[atk_action] for tar in range(self.env.n_targets)])
                # atk_prob = torch.stack([self.ppos[0].evaluate(self._get_atk_ob(tar, self.env.belief, states[0]), pre_rnns[0], actions[0], one_hot(self.env.n_targets, tar), tar, in_training=True)[1].detach() for tar in range(self.env.n_targets)])

                
                states, rew, done, _ = self.env.step(actions, atk_prob, verbose=True)
                # print('atk_strategy:')
                # # print(atk_strategy)
                # for type_i in range(self.env.n_types):
                #     _, _, atk_t_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, pre_rnns[0], type_n=type_i)
                #     print(atk_t_strategy)
                # print('def_strategy:')
                # print(def_strategy)

                atk_rew += rew[0]
                def_rew += rew[1]

                cur_round += 1
            atk_rews[cur_type] += atk_rew
            def_rews[cur_type] += def_rew

        for type_i in range(self.env.n_types):
            if type_cnt[type_i] != 0:
                print('type {}: sampled {} times, atk_avg_rew: {}, def_avg_rew: {}'.format(type_i, type_cnt[type_i], atk_rews[type_i] / type_cnt[type_i], def_rews[type_i] / type_cnt[type_i]))
    