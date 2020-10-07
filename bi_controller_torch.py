
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
from agent.bi_torch import NPAAgent, Memory, AtkNPAAgent
# from base_controller import BaseController
from env.base_env import BaseEnv
from agent.npa_torch import device
from datetime import datetime

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
    def __init__(self, env: BaseEnv, max_episodes, lr, betas, gamma, clip_eps, n_steps, network_width, test_every, n_belief, batch_size, minibatch, k_epochs=1000, max_process=3, v_batch_size=100000, seed=None):
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
        self.v_update_timestep = v_batch_size
        self.minibatch = minibatch
        self.lr = lr
        self.betas = betas
        self.gamma = gamma                # discount factor
        self.K_epochs = k_epochs                # update policy for K epochs
        self.eps_clip = clip_eps              # clip parameter for PPO
        self.random_seed = seed
        self.n_belief = n_belief
        self.max_process = max_process
        #############################################
        
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            # env.seed(random_seed)
        
        self.atk_memorys = [Memory() for _ in range(self.env.n_types)]
        self.def_memory = Memory()
        # self.def_memory
        self.ppos = []
        # print(self.env.observation_spaces[0])
        # print(self.env.observation_spaces[0].shape)
        # print(self.env.action_spaces[0])
        # print(self.env.action_spaces[0].n)

        # self.beliefs = [[self.generate_prior() for i in range(self.n_belief)] for j in range(self.env.n_steps)]
        # self.beliefs = [self.generate_belief(self.n_belief) for j in range(self.env.n_steps)]
        # self.beliefs = [[np.array([(1. / self.n_belief) * i, 1 - (1. / self.n_belief) * i]) for i in range(self.n_belief)] for j in range(self.env.n_steps)]
        # print('beliefs:')
        # print(self.beliefs)
        # atk_belief_n = []
        # for i in range(self.env.n_targets):
        #     new_belief = np.zeros(self.env.n_targets)
        #     new_belief[i] = 1
        #     atk_belief_n.append(np.copy(new_belief))
        # self.atk_belief = [atk_belief_n for i in range(self.env.n_steps)]
        # print('atk belief:')
        # print(self.atk_belief)
        # self.beliefs_n = [self.atk_belief, self.beliefs]
        # self.n_belief_n = [self.env.n_targets, self.n_belief]
        self.env_prior = np.copy(self.env.prior)
        self.atk_prior = np.array([(1. / self.env.n_targets) for _ in range(self.env.n_targets)])

        for i in range(env.n_agents):
            curmemory = []
            curppo = []
            # for j in range(env.n_steps):
            # memory = Memory()
            state_dim = self.env.observation_spaces[i].shape[0]
            action_dim = self.env.action_spaces[i].n
            if i != 0:
                ppo = NPAAgent(self.env.n_steps, state_dim, action_dim, self.n_latent_var, lr, betas[i], gamma, self.K_epochs, self.eps_clip, self.minibatch, self.env.n_targets)
            else:
                ppo = AtkNPAAgent(self.env.n_types, self.env.n_steps, state_dim, action_dim, self.n_latent_var, lr, betas[i], gamma, self.K_epochs, self.eps_clip, self.minibatch, self.env.n_targets)
                # curmemory.append(memory)
                # curppo.append(ppo)
            # self.memorys.append(memory)
            
            self.ppos.append(ppo)
        # print(lr,betas)
        
    
    def generate_prior(self):
        x = [0.] + sorted(np.random.rand(self.env.n_targets - 1).tolist()) + [1.]
        prior = np.zeros(self.env.n_targets)
        for i in range(self.env.n_targets):
            prior[i] = x[i + 1] - x[i]
        return prior
    
    def generate_belief(self, num):
        num_each_dim = int(num ** (1 / (self.env.n_targets - 1)))
        print('num each dim:')
        print(num_each_dim)
        step_len = 1 / num_each_dim
        print(step_len)
        
        # def dfs(r, )

    def _get_atk_ob(self, target, belief, last_obs_atk):
        if type(last_obs_atk) == torch.Tensor:
            ret = torch.Tensor(last_obs_atk)
            ret = ret[1 + self.env.n_targets: -self.env.n_targets]
            ret = torch.stack([torch.cat((belief, ret, torch.Tensor(one_hot(self.env.n_targets, target))))])
        else:
            ret = np.copy(last_obs_atk)
            # print(ret.shape)
            ret = ret[1 + self.env.n_targets : -self.env.n_targets]
            ret = np.array([np.concatenate((belief, ret, one_hot(self.env.n_targets, target)))])
        # print('ret shape:')
        # print(ret.shape)
        return ret    
    
    def _get_atk_ob_full(self, target, belief, last_obs_atk):
        if type(last_obs_atk) == torch.Tensor:
            ret = torch.Tensor(last_obs_atk)
            cur_round = ret[0]
            ret = ret[1 + self.env.n_targets: -self.env.n_targets]

            ret = torch.stack([torch.cat((torch.Tensor([cur_round]), belief, ret, torch.Tensor(one_hot(self.env.n_targets, target))))])
        else:
            ret = np.copy(last_obs_atk)
            # print(ret.shape)
            cur_round = ret[0]
            ret = ret[1 + self.env.n_targets: -self.env.n_targets]
            ret = np.array([np.concatenate((torch.Tensor([cur_round]), belief, ret, one_hot(self.env.n_targets, target)))])
        # print('ret shape:')
        # print(ret.shape)
        return ret    
    
        # training loop
    def train(self, num_round=None, round_each_belief = 1000, policy_store_every=100, 
               sec_prob=False, save_every=None, save_path=None, load_state=None, load_path=None, store_results=False):
        self.step = 0
        env = self.env
        if num_round == None:
            num_round = self.max_episodes
        
        running_reward = torch.zeros((self.env.n_targets, self.env.n_agents))
        avg_length = np.zeros((self.env.n_targets))
        timestep = 0
        epi_cnt = 0
        epi_type_cnt = np.zeros((self.env.n_targets))
        done_cnt = 0

        while self.step < num_round:
            self.step += 1
            for substep in range(self.env.n_steps - 1, -1, -1):
                print('start training substep {}.'.format(substep))

                # for b in range(self.n_belief):
                if True:
                    update_agent_num = 0
                    done_cnt = 0
                    for i_episode in range(round_each_belief):
                        if True:
                            if update_agent_num == 0:
                                states, _, _ = self.env.sub_reset(substep, self.atk_prior)
                            else:
                                curbelief = np.random.rand(self.env.n_types)
                                curbelief = curbelief / np.sum(curbelief)
                                # print(curbelief)
                                states, _, _ = self.env.sub_reset(substep, curbelief)
                        else:
                            states, _, _ = self.env.reset()

                        # curreward = torch.zeros((self.env.n_agents))
                        type_ob = np.zeros((1, self.env.n_targets))
                        type_ob[0, self.env.atk_type] = 1.
                        type_ob = torch.from_numpy(type_ob).float().to(device)

                        for t in range(substep, self.env.n_steps + 10):
                            # timestep += 1
                            curstep = substep
                            
                            actions = []
                            action, _ = self.ppos[0].act(curstep, states[0], self.atk_memorys[self.env.atk_type], self.env.atk_type, in_training=(t==substep))
                            actions.append(action)
                            action, _ = self.ppos[1].act(curstep, states[1], self.def_memory, in_training=(t==substep))
                            actions.append(action)
                            v = []
                            if t != substep:
                                # for i in range(self.env.n_agents):
                                # print('state:')
                                # print(states[0][1:], np.array([states[0][1:]]))
                                v.append(self.ppos[0].evaluate(t, torch.stack([states[0][1:]]), actions[0], self.env.atk_type, type_ob,in_training=False)[1])
                                v.append(self.ppos[1].evaluate(t, torch.stack([states[1][1:]]), actions[1], type_ob, in_training=False)[1])
                                done = True
                                reward = torch.Tensor(v)
                            # print('actions')
                            else:
                                atk_prob = torch.stack([self.ppos[0].evaluate(t, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, np.array([one_hot(self.env.n_targets, tar)]), in_training=t==substep)[3] for tar in range(self.env.n_targets)])
                                # with torch.no_grad():
                                states, reward, done, _ = self.env.step(actions, atk_prob)
                                reward = torch.Tensor(reward)
                            
                            # Saving reward and is_terminal:
                            # typeob is a tensor with shape[1, :, :], sue type_ob to extract the only episode in the type_ob
                            self.def_memory.rewards.append(reward[1])
                            self.def_memory.is_terminals.append(done)
                            self.def_memory.type_obs.append(type_ob[0])

                            self.atk_memorys[self.env.atk_type].rewards.append(reward[0])
                            self.atk_memorys[self.env.atk_type].is_terminals.append(done)
                            self.atk_memorys[self.env.atk_type].type_obs.append(type_ob[0])

                            if done:
                                done_cnt += 1
                                if done_cnt % self.update_timestep == 0 or i_episode == round_each_belief - 1:
                                    # for i in range(self.env.n_agents):
                                    # print('done cnt:')
                                    # print(done_cnt)
                                    print('{}: episode {} start training.'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), i_episode), end = ' ')
                                    
                                    if update_agent_num == 1:
                                        tot_loss = self.ppos[1].update(substep, self.def_memory)
                                    else:
                                        cntupd = 0
                                        v_loss, tot_loss = 0, 0
                                        for type_i in range(len(self.atk_memorys)):
                                            if len(self.atk_memorys[type_i].rewards) > 1:
                                                tot_loss_t = self.ppos[0].update(substep, self.atk_memorys[type_i], type_i)
                                                cntupd += 1
                                                # v_loss += v_loss_t
                                                tot_loss += tot_loss_t
                                        # v_loss /= cntupd
                                        tot_loss /= cntupd
                                    # v_loss += v_loss_t
                                    # tot_loss += tot_loss_t
                                    self.def_memory.clear_memory()
                                    
                                    for type_i in range(len(self.atk_memorys)):
                                        self.atk_memorys[type_i].clear_memory()
                                    print('{}: updated. loss: {:.4f}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), tot_loss))
                                    update_agent_num = (update_agent_num + 1) % 2
                                    # done_cnt = 0
                                # timestep = 0
                                break

                    for i_episode in range(2 * self.v_update_timestep):
                        if True:
                            if update_agent_num == 0:
                                states, _, _ = self.env.sub_reset(substep, self.atk_prior)
                            else:
                                curbelief = np.random.rand(self.env.n_types)
                                curbelief = curbelief / np.sum(curbelief)
                                # print(curbelief)
                                states, _, _ = self.env.sub_reset(substep, curbelief)
                        else:
                            states, _, _ = self.env.reset()

                        # curreward = torch.zeros((self.env.n_agents))
                        type_ob = np.zeros((1, self.env.n_targets))
                        type_ob[0, self.env.atk_type] = 1.
                        type_ob = torch.from_numpy(type_ob).float().to(device)

                        for t in range(substep, self.env.n_steps + 10):
                            # timestep += 1
                            curstep = substep
                            
                            actions = []

                            action, _ = self.ppos[0].act(curstep, states[0], self.atk_memorys[self.env.atk_type], self.env.atk_type, in_training=True)
                            actions.append(action)
                            action, _ = self.ppos[1].act(curstep, states[1], self.def_memory, in_training=True)
                            actions.append(action)
                            v = []
                            if t != substep:
                                v.append(self.ppos[0].evaluate(t, torch.stack([states[0][1:]]), actions[0], self.env.atk_type, type_ob,  in_training=False)[1])
                                v.append(self.ppos[1].evaluate(t, torch.stack([states[1][1:]]), actions[1], type_ob,  in_training=False)[1])
                                done = True
                                reward = torch.Tensor(v)
                            else:
                                atk_prob = torch.stack([self.ppos[0].evaluate(t, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, np.array([one_hot(self.env.n_targets, tar)]), in_training=True)[3] for tar in range(self.env.n_targets)])
                                states, reward, done, _ = self.env.step(actions, atk_prob)
                                reward = torch.Tensor(reward)

                            # if not done and t != substep:
                            #     done = True
                            #     # print('v:')
                            #     # print(v)
                            #     reward = torch.Tensor(v)
                            # else:
                            #     reward = torch.Tensor(reward)
                            
                            # Saving reward and is_terminal:
                            # typeob is a tensor with shape[1, :, :], sue type_ob to extract the only episode in the type_ob
                            self.def_memory.rewards.append(reward[1])
                            self.def_memory.is_terminals.append(done)
                            self.def_memory.type_obs.append(type_ob[0])

                            self.atk_memorys[self.env.atk_type].rewards.append(reward[0])
                            self.atk_memorys[self.env.atk_type].is_terminals.append(done)
                            self.atk_memorys[self.env.atk_type].type_obs.append(type_ob[0])
                            
                            # if substep == 0:
                            #     curreward += reward

                            if done:
                                done_cnt += 1
                                if done_cnt % self.v_update_timestep == 0 or i_episode == round_each_belief - 1:
                                    print('{}: episode {} start training.'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), i_episode), end = ' ')
                                    
                                    if update_agent_num == 1:
                                        v_loss = self.ppos[1].v_update(substep, self.def_memory)
                                    else:
                                        cntupd = 0
                                        v_loss, tot_loss = 0, 0
                                        for type_i in range(len(self.atk_memorys)):
                                            if len(self.atk_memorys[type_i].rewards) > 1:
                                                v_loss_t = self.ppos[0].v_update(substep, self.atk_memorys[type_i], type_i)
                                                cntupd += 1
                                                v_loss += v_loss_t
                                                # tot_loss += tot_loss_t
                                        v_loss /= cntupd
                                        # tot_loss /= cntupd

                                    self.def_memory.clear_memory()
                                    
                                    for type_i in range(len(self.atk_memorys)):
                                        self.atk_memorys[type_i].clear_memory()
                                    print('{}: updated. V_loss: {:.4f}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), v_loss, ))
                                    update_agent_num = (update_agent_num + 1) % 2
                                break
            atk_dict = self.ppos[0].get_state_dict()
            def_dict = self.ppos[1].get_state_dict()
            # print(os.path.exists('models'))
            # print(len(atk_dict))
            # print(len(atk_dict[0]))
            save_model(atk_dict, 'models/atk_bi_model_r_{}_bsize_{}_time_{}.pickle'.format(round_each_belief, self.update_timestep, datetime.now().strftime("%m:%d,%H:%M:%S")))
            save_model(def_dict, 'models/def_bi_model_r_{}_bsize_{}_time_{}.pickle'.format(round_each_belief, self.update_timestep, datetime.now().strftime("%m:%d,%H:%M:%S")))