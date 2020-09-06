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
from agent.sample_ac import NPAAgent, AtkNPAAgent
from env.base_env import BaseEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def one_hot(n, i):
    x = np.zeros(n)
    if i < n:
        x[i] = 1.
    return x

class NaiveController():
    def __init__(self, env: BaseEnv, max_episodes, lr, betas, gamma, clip_eps, n_steps, network_width, test_every, n_belief, n_states, batch_size, minibatch, k_epochs, seed=None):
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
        print('atk belief:')
        print(self.atk_belief)
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
                ppo = NPAAgent(self.env.n_steps, self.n_belief, self.beliefs, state_dim, action_dim, self.n_latent_var, lr, betas[i], gamma, self.K_epochs)
            else:
                ppo = AtkNPAAgent(self.env.n_types, self.env.n_steps, self.n_belief, self.beliefs, state_dim, action_dim, self.n_latent_var, lr, betas[i], gamma, self.K_epochs)
            # self.memorys.append(memory)
            self.agents.append(ppo)
    
    def _get_atk_ob(self, target, belief, last_obs_atk) :
        ret = np.copy(last_obs_atk)
        ret = ret[1 + self.env.n_targets * 2:]
        ret = np.array([np.concatenate((one_hot(self.env.n_targets, target), belief, ret))])
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

                        print('start training in round {}, belief {}, state {}'.format(substep, b, i_state))

                        # update the agents
                        for i_episode in range(round_each_belief):
                            # if update_agent_num == 1 or i_episode == 0:
                            def_vs = []
                            atk_vs = [[] for _ in range(self.env.n_types)]
                            for atk_action in range(self.action_dims[0]):
                                # need to get atk_prob
                                # TODO update return and API
                                atk_prob = [self.agents[0].evaluate(substep, self._get_atk_ob(tar, self.env.belief, train_state[0]), atk_action, b, self.env.atk_type)[0][0].cpu().detach().numpy() for tar in range(self.env.n_targets)]
                                # print('atk prob:')
                                # print(atk_prob)
                                def_tmpvs = []
                                atk_tmpvs = [[] for _ in range(self.env.n_types)]
                                for def_action in range(self.action_dims[1]):
                                    # sumv = None
                                    def_sumv = 0
                                    atk_sumv = [0 for _ in range(self.env.n_types)]
                                    type_cnts = [0 for _ in range(self.env.n_types)]
                                    actions = [atk_action, def_action]
                                    
                                    for type_i in range(self.env.n_types):
                                        self.env.reset_to_state_with_type(state_info, belief, type_i)
                                        next_states, reward, done, _ = self.env.step(actions, atk_prob)
                                        tmpv = reward
                                        if not done:
                                            # tmpv[0] += self.gamma * self.agents[0].evaluate(substep + 1, next_states[0], b, type_i)[1]
                                            tmpv[1] += self.gamma * self.agents[1].evaluate(substep + 1, next_states[1], b)[1]
                                        type_cnts[self.env.atk_type] += 1
                                        atk_sumv[self.env.atk_type] += tmpv[1]
                                        # def_sumv += tmpv[0]
                                    self.env.reset_to_state(state_info, belief)

                                    # reward = []
                                    for epoch_i in range(self.K_epochs):
                                        
                                        next_states, reward, done, _ = self.env.step(actions, atk_prob)
                                        tmpv = reward
                                        if not done:
                                            tmpv[0] += self.gamma * self.agents[0].evaluate(substep + 1, next_states[0], b, type_i)[1]
                                            tmpv[1] += self.gamma * self.agents[1].evaluate(substep + 1, next_states[1], b)[1]
                                        type_cnts[self.env.atk_type] += 1
                                        atk_sumv[self.env.atk_type] += tmpv[1]
                                        def_sumv += tmpv[0]
                                        self.env.reset_to_state(state_info, belief)
                                    # print(sumv)
                                    # tmpvs.append(sumv / self.K_epochs)
                                    def_tmpvs.append(def_sumv / self.K_epochs)
                                    # atk_tmpvs.append(sumv[0] / self.K_epochs)
                                    for type_i in range(self.env.n_types):
                                        atk_tmpvs[type_i].append(atk_sumv[type_i] / type_cnts[type_i])
                                    # self.agents[update_agent_num]
                                # vs.append(tmpvs)
                                def_vs.append(def_tmpvs)
                                # atk_vs.append(atk_tmpvs)
                                for type_i in range(self.env.n_types):
                                    atk_vs[type_i].append(atk_tmpvs[type_i])
                            # vs = np.array(vs)
                            def_vs = np.array(def_vs)
                            atk_vs = np.array(atk_vs)
                            # atk_s = np.zeros(self.env.n_types)
                            atk_s = None
                            for i in range(self.env.n_types):
                                if i == 0:
                                    atk_s = self.agents[0].act(substep, train_state[0], b, i)[1].detach() * self.beliefs[substep][b][i]
                                else:
                                    atk_s += self.agents[0].act(substep, train_state[0], b, i)[1].detach() * self.beliefs[substep][b][i]
                            
                            # print('vs:')
                            # print(def_vs, atk_vs)
                            # print(type(atk_vs))
                            # print(atk_vs.shape)

                            self.agents[1].update(substep, train_state[1], atk_s, b, def_vs.T)
                            self.agents[0].update(substep, train_state[0], self.agents[1].act(substep, train_state[1], b)[1].detach(), b, atk_vs)
                        
                        # calculate the value function
                        def_value = 0
                        def_strategy = self.agents[1].act(substep, train_state[1], b)[1].detach()

                        for type_i in range(self.env.n_types):
                            ev_atk = 0
                            ev_def = 0
                            train_state,  _, _ = self.env.reset_to_state_with_type(state_info, belief, type_i)
                            atk_strategy = self.agents[0].act(substep, train_state[0], b, type_i)[1].detach()
                            atk_obs[type_i].append(train_state[0][1 + self.env.n_targets:])
                            
                            for atk_action in range(self.action_dims[0]):
                                # need to get atk_prob
                                # TODO update return and API
                                atk_prob = [self.agents[0].evaluate(substep, self._get_atk_ob(tar, self.env.belief, train_state[0]), atk_action, b, self.env.atk_type)[0].cpu().detach().numpy() for tar in range(self.env.n_targets)]
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
                                            tmpv[0] += self.gamma * self.agents[0].evaluate(substep + 1, next_states[0], b, type_i)[1]
                                            tmpv[1] += self.gamma * self.agents[1].evaluate(substep + 1, next_states[1], b)[1]
                                        if epoch_i > 0:
                                            sumv += tmpv
                                        else:
                                            sumv = np.array(tmpv)
                                    sumv /= self.K_epochs
                                    ev_atk += sumv[0] * atk_strategy[atk_action] * def_strategy[def_action]
                                    ev_def += sumv[1] * atk_strategy[atk_action] * def_strategy[def_action]
                            # print('def value:')
                            # print(ev_def, self.beliefs[substep][b][type_i], def_value)
                            def_value += ev_def * self.beliefs[substep][b][type_i]
                            # atk_values.append(ev_atk)
                            atk_value_preds[type_i].append(ev_atk.detach())
                            
                            atk_p_res[type_i].append(self.agents[0].act(substep, train_state[0], b, type_i)[1].detach())
                        def_value_preds.append(def_value.detach())
                        def_p_res.append(def_strategy.detach())


                    self.agents[0].value_supervise(substep, b, atk_obs, atk_value_preds)
                    self.agents[0].policy_supervise(substep, b, atk_obs, atk_p_res)
                    self.agents[1].value_supervise(substep, b, def_obs, def_value_preds)
                    self.agents[1].policy_supervise(substep, b, def_obs, def_p_res)


                            # # update_agent_num = int(i_episode / self.update_timestep) % 2
                            # if update_agent_num == 0:
                            #     belief_init = self.atk_prior
                            #     start_num = self.atk_type

                            #     # first train value
                            #     for atk_action in range(self.action_dims[0]):
                            #         atk_prob = [self.ppos[0].evaluate(t, self._get_atk_ob(tar, self.env.belief, train_state), atk_action, self.env.atk_type, type_ob)[3].cpu().detach().numpy() for tar in range(self.env.n_targets)]
                            #         for def_action in range(self.action_dims[1]):
                            #             actions_comb = [atk_action, def_action]
                            #             n_state, rew, done, _ = self.env.step(action_comb, atk_prob)
                            #             value_preds.append(rew + self.gamma * )

                            #             train_state = self.env.reset_to_state(state_info, belief)

                            #     # self.env.set_prior(self.atk_prior)
                            # else:
                            #     belief_init = self.beliefs[substep][b]
                            #     start_num = b
                            #     # self.env.set_prior(self.env_prior)

                            # # if substep != 0:
                            #     # states, _, _ = self.env.sub_reset(substep, belief_init)
                            # # else:
                            #     # states, _, _ = self.env.reset()
                            # states, _, _ = self.env.reset_to_state(state_info, belief)
                            # oppo_strategy = self.agents[oppo_agent_num].act(substep, states)

                            # for _ in range(self.K_epochs):
                            #     self.agents[update_agent_num].update(substep, states, oppo_strategy, )
                            
                            
                            # update_agent_num = (update_agent_num + 1) % 2
                            # oppo_agent_num = (oppo_agent_num + 1) % 2