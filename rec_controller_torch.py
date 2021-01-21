
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
        self.update_timestep = 100      # update policy every n timesteps
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
        
        self.memorys = []
        self.ppos = []
        # print(self.env.observation_spaces[0])
        print(self.env.observation_spaces[0].shape)
        print(self.env.observation_spaces[1].shape)
        # print(self.env.action_spaces[0])
        # print(self.env.action_spaces[0].n)

        for i in range(env.n_agents):
            # curmemory = []
            curppo = []
            # for j in range(env.n_steps):
            # memory = Memory()
            state_dim = self.env.observation_spaces[i].shape[0]
            action_dim = self.env.action_spaces[i].n
            if i > 0:
                ppo = PPO(state_dim + self.env.n_targets, action_dim, self.n_latent_var, lr, betas[i], gamma, self.K_epochs, self.eps_clip, self.env.n_targets)
            else:
                ppo = AtkNPAAgent(self.env.n_targets, state_dim + self.env.n_targets, action_dim, self.n_latent_var, lr, betas[i], gamma, self.K_epochs, self.eps_clip, self.env.n_targets)
                # curmemory.append(memory)
                # curppo.append(ppo)
            # self.memorys.append(memory)
            self.ppos.append(ppo)
        # print(lr,betas)

        self.atk_memorys = [Memory() for _ in range(self.env.n_types)]
        self.def_memory = Memory()

        self.env_prior = np.copy(self.env.prior)
        self.atk_prior = np.array([(1. / self.env.n_targets) for _ in range(self.env.n_targets)])
    
    def load_models(self, exp_name):
        atk_dict_path = 'models/atk_{}.pickle'.format(exp_name)
        atk_dict_to_load = load_model(atk_dict_path)
        def_dict_path = 'models/def_{}.pickle'.format(exp_name)
        def_dict_to_load = load_model(def_dict_path)

        # print(len(atk_dict_to_load))
        # print(len(atk_dict_to_load[0]))

        self.ppos[0].set_state_dict(atk_dict_to_load)
        self.ppos[1].set_state_dict(def_dict_to_load)
    
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
                pre_rnns = []

                for i in range(self.env.n_agents):
                    # pre_rnns.append(np.copy(rnn_historys[i]))
                    pre_rnns.append(rnn_historys[i])
                    # action = self.ppos[i].act(curstep, states[i], self.memorys[i])
                    if i == 0:
                        action, rnn_historys[i], _ = self.ppos[i].act(states[i], self.atk_memorys[self.env.atk_type], rnn_historys[i], self.env.atk_type)
                    else:
                        action, rnn_historys[i], _ = self.ppos[i].act(states[i], self.def_memory, rnn_historys[i])
                    actions.append(action)
                
                atk_prob = torch.stack([self.ppos[0].evaluate(self._get_atk_ob(tar, self.env.belief, states[0]), pre_rnns[0], actions[0], one_hot(self.env.n_targets, tar), tar, in_training=True)[1].detach() for tar in range(self.env.n_targets)])
                # atk_prob = 0
                with torch.no_grad():
                    states, reward, done, _ = env.step(actions, atk_prob)
                
                # Saving reward and is_terminal:
                # for i in range(self.env.n_agents):
                #     self.memorys[i].rewards.append(reward[i])
                #     self.memorys[i].is_terminals.append(done)
                #     self.memorys[i].type_obs.append(type_ob[0])
                
                self.def_memory.rewards.append(reward[1])
                self.def_memory.is_terminals.append(done)
                self.def_memory.type_obs.append(type_ob[0])

                self.atk_memorys[self.env.atk_type].rewards.append(reward[0])
                self.atk_memorys[self.env.atk_type].is_terminals.append(done)
                self.atk_memorys[self.env.atk_type].type_obs.append(type_ob[0])

                # update if its time
                if done and self.step % self.update_timestep == 0:
                    # print('did update')
                    train_agent_n = int(self.step / self.update_timestep) % 2
                    # for i in range(self.env.n_agents):
                    # v_loss = self.ppos[train_agent_n].update(self.memorys[train_agent_n])
                    v_loss = 0
                    if train_agent_n == 0:
                        for type_i in range(self.env.n_types):
                            v_loss += self.ppos[0].update(self.atk_memorys[type_i], type_i)
                    else:
                        v_loss = self.ppos[1].update(self.def_memory)
                    # for agent_i in range(self.env.n_agents):
                        # self.memorys[agent_i].clear_memory()
                    for type_i in range(self.env.n_types):
                        self.atk_memorys[type_i].clear_memory()
                    self.def_memory.clear_memory()
                    # if timestep % 
                    print('timestep{} updated with q loss{}'.format(timestep, v_loss))
                    # timestep = 0
                    # print('')
                
                running_reward += reward

                if done:
                    # for i in range(self.env.n_agents):
                        # self.ppos[i].update(self.memorys[i])
                        # self.memorys[i].clear_memory()
                    # timestep = 0
                    # print('done!!!!')
                    break

            avg_length += current_len
            current_len = 0

        atk_dict = self.ppos[0].get_state_dict()
        def_dict = self.ppos[1].get_state_dict()
        # print(os.path.exists('models'))
        # print(len(atk_dict))
        # print(len(atk_dict[0]))
        save_model(atk_dict, 'models/atk_rnn_model_r_{}_bsize_{}_time_{}.pickle'.format(num_round, self.update_timestep, datetime.now().strftime("%m:%d,%H:%M:%S")))
        save_model(def_dict, 'models/def_rnn_model_r_{}_bsize_{}_time_{}.pickle'.format(num_round, self.update_timestep, datetime.now().strftime("%m:%d,%H:%M:%S")))
            
            
            # logging
            # if self.step % self.log_interval == 0:

            #     avg_length = int(avg_length/self.log_interval)
            #     running_reward /= self.log_interval
            #     # running_reward = int((running_reward/self.log_interval))

                
            #     # print('Episode {} \t episode length: {} \t reward:'.format(self.step, avg_length))
            #     # print(running_reward)
            #     running_reward = np.zeros((self.env.n_agents))
            #     avg_length = 0
    
    def sub_game_exploitability(self, single_train_round=50000, episodes_test=50, exp_name=None):
        if exp_name != None:
            self.load_models(exp_name)
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
                atk_action, rnn_historys[0], atk_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, rnn_historys[0], type_n=self.env.atk_type)
                actions.append(atk_action)
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
                atk_prob = torch.stack([self.ppos[0].evaluate(self._get_atk_ob(tar, self.env.belief, states[0]), pre_rnns[0], actions[0], one_hot(self.env.n_targets, tar), tar, in_training=True)[1].detach() for tar in range(self.env.n_targets)])
                # atk_prob = 0
                # with torch.no_grad():
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
                    # print('timestep{} updated with q loss{}'.format(timestep, v_loss))
                    # timestep = 0
                
                running_reward += reward

                if done:
                    # for i in range(self.env.n_agents):
                        # self.ppos[i].update(self.memorys[i])
                        # self.memorys[i].clear_memory()
                    # timestep = 0
                    break
        
        # atk_rews = [0 for _ in range(self.env.n_types)]
        # def_rews = [0 for _ in range(self.env.n_types)]
        # type_cnt = [0 for _ in range(self.env.n_types)]
        
        full_defs = []
        
        full_atks = [[] for _ in range(2)]

        for atk_a in range(4):
            for def_a in range(4):
                for atk_a_2 in range(4):
                    for def_a_2 in range(4):
                        
                        cur_max_def = [-np.inf for _ in range(self.env.n_types)]
                        cur_max_atk = [-np.inf for _ in range(self.env.n_types)]
                        def_strategies = None
                        type_possis = None

                        for atk_a_3 in range(4):
                            cur_def_rew = [0 for _ in range(self.env.n_types)]
                            cur_atk_rew = [0 for _ in range(self.env.n_types)]

                            for def_a_3 in range(5):
                                atk_rews = [0 for _ in range(self.env.n_types)]
                                def_rews = [0 for _ in range(self.env.n_types)]
                                type_cnt = [0 for _ in range(self.env.n_types)]

                                for i in range(episodes_test):
                                    cur_round = 0
                                    # states, _, _ = self.env.reset()
                                    print('new episode:')
                                    states, _, _ = self.env.reset_to_state(cur_round, (np.array([0, 4]), np.array([1, 4])), torch.Tensor(self.env.prior))
                                    # states, _, _ = self.env.sub_reset(0, self.env.prior)
                                    cur_type = self.env.atk_type
                                    done = False
                                    atk_rew = 0
                                    def_rew = 0
                                    rnn_historys = [torch.from_numpy(np.zeros((1, 1, self.n_latent_var))).float().to(device) for _ in range(self.env.n_agents)]
                                    is_first_round=True
                                    is_second_round=True
                                    is_third_round = True
                                    need_get_type=False
                                    if def_strategies is not None and def_strategies[def_a_3] < 1e-6:
                                        break
                                    while not done:
                                        pre_rnns = []
                                        pre_rnns.append(rnn_historys[0])
                                        pre_rnns.append(rnn_historys[1])

                                        # atk_action, rnn_historys[0], atk_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, rnn_historys[0], type_n=cur_type)
                                        
                                        if is_first_round:
                                            actions = [atk_a, def_a]
                                            is_first_round = False
                                        elif is_second_round:
                                            actions = [atk_a_2, def_a_2]
                                            is_second_round = False
                                        elif is_third_round:
                                            actions = [atk_a_3, def_a_3]
                                            is_third_round = False
                                            need_get_type=True
                                            if def_strategies is None:
                                                _, def_strategies = train_agent.act(cur_round, states[1], train_memory)
                                                type_possis = self.env.get_cur_state()[1]
                                        else:
                                            # atk_action, atk_strategy = self.ppos[0].act(cur_round, states[0], self.ppos[0].memory_ph, -1,  type_n=cur_type)
                                            # def_action, def_strategy = train_agent.act(cur_round, states[1], train_memory)
                                            atk_action, rnn_historys[0], atk_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, rnn_historys[0], type_n=cur_type)
                                            def_action, def_strategy = train_agent.act(cur_round, states[1], train_memory)
                                            # atk_prob = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), atk_action, tar, np.array([one_hot(self.env.n_targets, tar)]))[1].detach() for tar in range(self.env.n_targets)])
                                            actions = [atk_action, def_action]
                                        
                                        atk_prob = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, np.array([one_hot(self.env.n_targets, tar)]), -1, in_training=False)[3] for tar in range(self.env.n_targets)])
                                        
                                        states, rew, done, _ = self.env.step(actions, atk_prob, verbose=True)

                                        # if not is_second_round and is_third_round and type(def_strategies) == NoneType:
                                            # def_strategies = self.env.

                                        if need_get_type:
                                            poss, belief = self.env.get_cur_state()
                                            belief = belief.detach()
                                            belief = belief / torch.sum(belief)
                                            states, _, _ = self.env.reset_to_state(cur_round + 1, poss, belief)
                                            cur_type = self.env.atk_type
                                            type_cnt[cur_type] += 1
                                            need_get_type = False
                                        # print('atk_strategy:')
                                        # # print(atk_strategy)
                                        # for type_i in range(self.env.n_types):
                                        #     _,  atk_t_strategy = self.ppos[0].act(cur_round, states[0], self.ppos[0].memory_ph, type_n=type_i)
                                        #     print(atk_t_strategy)
                                        # print('def_strategy:')
                                        # print(def_strategy)

                                        atk_rew += rew[0]
                                        def_rew += rew[1]

                                        cur_round += 1
                                    atk_rews[cur_type] += atk_rew
                                    def_rews[cur_type] += def_rew
                                    type_cnt[cur_type] += 1
                                # tot_rew_atk = 0
                                # tot_rew_def = 0
                                # for type_i in range(self.env.n_types):
                                    # if atk_rew[type_i] / type_cnt[type_i] > cur_max:
                                        # cur_max = atk_rew[type_i]
                                    # tot_rew_atk += atk_rew[type_i] # / type_cnt[type_i]
                                    # tot_rew_def += def_rew[type_i]
                                # if tot_rew_atk > cur_max_atk:
                                #     cur_max_atk = tot_rew_atk
                                # if tot_rew_def > cur_max_def:
                                #     cur_max_def = tot_rew_def
                                for type_i in range(self.env.n_types):
                                    if type_cnt[type_i] > 0:
                                        print(def_strategies, def_rews, type_cnt)
                                        cur_def_rew[type_i] += def_strategies[def_a_3] * def_rews[type_i] / type_cnt[type_i]
                                        cur_atk_rew[type_i] += def_strategies[def_a_3] * atk_rews[type_i] / type_cnt[type_i]
                            for type_i in range(self.env.n_types):
                                if cur_atk_rew[type_i] > cur_max_atk[type_i]:
                                    cur_max_atk[type_i] = cur_atk_rew[type_i]
                                    cur_max_def[type_i] = cur_def_rew[type_i]    

                        # full_defs.append((def_rews[0] + def_rews[1]) * 1./ (type_cnt[0] + type_cnt[1]))
                        full_defs.append(cur_max_def[0] * type_possis[0] + cur_max_def[1] * type_possis[1])
                        full_atks[0].append(cur_max_atk[0])
                        full_atks[1].append(cur_max_atk[1])

        # full_defs = []
        # full_atks = [[] for _ in range(2)]

        # # first_round_actions = []
        # for atk_a in range(4):
        #     for def_a in range(4):
        #         for atk_a_2 in range(4):
        #             for def_a_2 in range(4):
        #                 atk_rews = [0 for _ in range(self.env.n_types)]
        #                 def_rews = [0 for _ in range(self.env.n_types)]
        #                 type_cnt = [0 for _ in range(self.env.n_types)]
        #                 for i in range(episodes_test):
        #                     cur_round = 0
        #                     # states, _, _ = self.env.reset()
        #                     states, _, _ = self.env.reset_to_state((np.array([0, 4]), np.array([1, 4])), self.env.prior)
        #                     print('new episode:, prior{}'.format(self.env.prior))
        #                     # states, _, _ = self.env.sub_reset(0, self.env.prior)
        #                     cur_type = self.env.atk_type
        #                     type_cnt[cur_type] += 1
        #                     done = False
        #                     atk_rew = 0
        #                     def_rew = 0
        #                     rnn_historys = [torch.from_numpy(np.zeros((1, 1, self.n_latent_var))).float().to(device) for _ in range(self.env.n_agents)]
        #                     is_first_round=True
        #                     is_second_round=True
        #                     need_get_type=False
        #                     while not done:
        #                         pre_rnns = []
        #                         pre_rnns.append(rnn_historys[0])
        #                         pre_rnns.append(rnn_historys[1])

        #                         if is_first_round:
        #                             actions = [atk_a, def_a]
        #                             is_first_round = False
        #                         elif is_second_round:
        #                             actions = [atk_a_2, def_a_2]
        #                             is_second_round = False
        #                             need_get_type = True
        #                         else:
        #                             atk_action, rnn_historys[0], atk_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, rnn_historys[0], type_n=cur_type)
        #                             def_action, def_strategy = train_agent.act(cur_round, states[1], train_memory)
        #                             # atk_prob = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), atk_action, tar, np.array([one_hot(self.env.n_targets, tar)]))[1].detach() for tar in range(self.env.n_targets)])
        #                             actions = [atk_action, def_action]
        #                         atk_prob = torch.stack([self.ppos[0].evaluate(self._get_atk_ob(tar, self.env.belief, states[0]), pre_rnns[0], actions[0], one_hot(self.env.n_targets, tar), tar, in_training=True)[1].detach() for tar in range(self.env.n_targets)])

                                
        #                         states, rew, done, _ = self.env.step(actions, atk_prob, verbose=True)
        #                         if need_get_type:
        #                             poss, belief = self.env.get_current_state()
        #                             states, _, _ = self.env.reset_to_state(poss, belief)
        #                             cur_type = self.env.atk_type
        #                             type_cnt[cur_type] += 1
        #                             need_get_type = False
        #                         # print('atk_strategy:')
        #                         # # print(atk_strategy)
        #                         # for type_i in range(self.env.n_types):
        #                         #     _, _, atk_t_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, pre_rnns[0], type_n=type_i)
        #                         #     print(atk_t_strategy)
        #                         # print('def_strategy:')
        #                         # print(def_strategy)

        #                         atk_rew += rew[0]
        #                         def_rew += rew[1]

        #                         cur_round += 1
        #                     atk_rews[cur_type] += atk_rew
        #                     def_rews[cur_type] += def_rew
        #                 full_defs.append((def_rews[0] + def_rews[1]) * 1./ (type_cnt[0] + type_cnt[1]))
        #                 if type_cnt[0] > 0:
        #                     full_atks[0].append(atk_rews[0] / type_cnt[0])
        #                 else:
        #                     full_atks[0].append(0.)
        #                 if type_cnt[1] > 0:
        #                     full_atks[1].append(atk_rews[1] / type_cnt[1])
        #                 else:
        #                     full_atks[1].append(0.)
        print(full_defs)
        print(torch.mean(full_defs))
        print(full_atks)
        print(torch.mean(full_atks[0], full_atks[1]))

        # for type_i in range(self.env.n_types):
        #     if type_cnt[type_i] != 0:
        #         print('type {}: sampled {} times, atk_avg_rew: {}, def_avg_rew: {}'.format(type_i, type_cnt[type_i], atk_rews[type_i] / type_cnt[type_i], def_rews[type_i] / type_cnt[type_i]))
    
    def calculate_exploitability(self, single_train_round=100000, episodes_test=100, exp_name=None):
        if exp_name != None:
            self.load_models(exp_name)
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
                atk_action, rnn_historys[0], atk_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, rnn_historys[0], type_n=self.env.atk_type)
                actions.append(atk_action)
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
                atk_prob = torch.stack([self.ppos[0].evaluate(self._get_atk_ob(tar, self.env.belief, states[0]), pre_rnns[0], actions[0], one_hot(self.env.n_targets, tar), tar, in_training=True)[1].detach() for tar in range(self.env.n_targets)])
                # atk_prob = 0
                # with torch.no_grad():
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
                    # print('timestep{} updated with q loss{}'.format(timestep, v_loss))
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

        # first_round_actions = []
        # for atk_a in range(4):
        #     for def_a in range(4):
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

                atk_action, rnn_historys[0], atk_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, rnn_historys[0], type_n=cur_type)
                def_action, def_strategy = train_agent.act(cur_round, states[1], train_memory)
                # atk_prob = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), atk_action, tar, np.array([one_hot(self.env.n_targets, tar)]))[1].detach() for tar in range(self.env.n_targets)])
                actions = [atk_action, def_action]
                atk_prob = torch.stack([self.ppos[0].evaluate(self._get_atk_ob(tar, self.env.belief, states[0]), pre_rnns[0], actions[0], one_hot(self.env.n_targets, tar), tar, in_training=True)[1].detach() for tar in range(self.env.n_targets)])

                
                states, rew, done, _ = self.env.step(actions, atk_prob, verbose=True)
                print('atk_strategy:')
                # print(atk_strategy)
                for type_i in range(self.env.n_types):
                    _, _, atk_t_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, pre_rnns[0], type_n=type_i)
                    print(atk_t_strategy)
                print('def_strategy:')
                print(def_strategy)

                atk_rew += rew[0]
                def_rew += rew[1]

                cur_round += 1
            atk_rews[cur_type] += atk_rew
            def_rews[cur_type] += def_rew

        for type_i in range(self.env.n_types):
            if type_cnt[type_i] != 0:
                print('type {}: sampled {} times, atk_avg_rew: {}, def_avg_rew: {}'.format(type_i, type_cnt[type_i], atk_rews[type_i] / type_cnt[type_i], def_rews[type_i] / type_cnt[type_i]))
    
    def assess_strategy(self, episodes_test=100):
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

                atk_action, rnn_historys[0], atk_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, rnn_historys[0], type_n=cur_type)
                def_action, rnn_historys[1], def_strategy = self.ppos[1].act(states[1], self.ppos[1].memory_ph, rnn_historys[1])
                # atk_prob = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), atk_action, tar, np.array([one_hot(self.env.n_targets, tar)]))[1].detach() for tar in range(self.env.n_targets)])
                actions = [atk_action, def_action]
                atk_prob = torch.stack([self.ppos[0].evaluate(self._get_atk_ob(tar, self.env.belief, states[0]), pre_rnns[0], actions[0], one_hot(self.env.n_targets, tar), tar, in_training=True)[1].detach() for tar in range(self.env.n_targets)])

                
                states, rew, done, _ = self.env.step(actions, atk_prob, verbose=True)
                print('atk_strategy:')
                # print(atk_strategy)
                for type_i in range(self.env.n_types):
                    _, _, atk_t_strategy = self.ppos[0].act(states[0], self.ppos[0].memory_ph, pre_rnns[0], type_n=type_i)
                    print(atk_t_strategy)
                print('def_strategy:')
                print(def_strategy)

                atk_rew += rew[0]
                def_rew += rew[1]
            atk_rews[cur_type] += atk_rew
            def_rews[cur_type] += def_rew

        for type_i in range(self.env.n_types):
            if type_cnt[type_i] != 0:
                print('type {}: sampled {} times, atk_avg_rew: {}, def_avg_rew: {}'.format(type_i, type_cnt[type_i], atk_rews[type_i] / type_cnt[type_i], def_rews[type_i] / type_cnt[type_i]))