
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
from agent.pg_npa import NPAAgent, AtkNPAAgent
# from base_controller import BaseController
from env.base_env import BaseEnv

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def one_hot(n, i):
    x = np.zeros(n)
    if i < n:
        x[i] = 1.
    return x

class NaiveController():
    def __init__(self, env: BaseEnv, max_episodes, lr, betas, gamma, clip_eps, n_steps, network_width, test_every, n_belief, batch_size, minibatch, seed=None):
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
        self.K_epochs = 100                # update policy for K epochs
        self.eps_clip = clip_eps              # clip parameter for PPO
        self.random_seed = seed
        self.n_belief = n_belief
        #############################################
        
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            # env.seed(random_seed)
        
        # self.atk_memorys = [Memory() for _ in range(self.env.n_types)]
        # self.def_memory = Memory()
        # self.def_memory
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

        for i in range(env.n_agents):
            curmemory = []
            curppo = []
            # for j in range(env.n_steps):
            # memory = Memory()
            state_dim = self.env.observation_spaces[i].shape[0]
            action_dim = self.env.action_spaces[i].n
            if i != 0:
                ppo = NPAAgent(self.env.n_steps, self.n_belief, self.beliefs, lr, betas[i], state_dim, action_dim, self.n_latent_var, self.env.n_targets)
            else:
                ppo = AtkNPAAgent(self.env.n_types, self.env.n_steps, self.n_belief, self.beliefs, lr, betas[i], state_dim, action_dim, self.n_latent_var, self.env.n_targets)
                # curmemory.append(memory)
                # curppo.append(ppo)
            # self.memorys.append(memory)
            
            self.agents.append(ppo)
        # print(lr,betas)
        print('print every:')
        print(self.log_interval)
    
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

    def _get_atk_ob(self, target, belief, last_obs_atk) :
        if type(last_obs_atk) == torch.Tensor:
            ret = torch.Tensor(last_obs_atk)
            ret = ret[1 + self.env.n_targets * 2:]
            ret = torch.stack([torch.cat((belief, ret, torch.Tensor(one_hot(self.env.n_targets, target))))])
        else:
            ret = np.copy(last_obs_atk)
            # print(ret.shape)
            ret = ret[1 + self.env.n_targets * 2:]
            ret = np.array([np.concatenate((belief, ret, one_hot(self.env.n_targets, target)))])
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
            for substep in range(env.n_steps - 1, -1, -1):
                print('start training substep {}.'.format(substep))

                for b in range(self.n_belief):
                    update_agent_num = 0
                    done_cnt = 0
                    for i_episode in range(round_each_belief):

                        # update_agent_num = int(i_episode / self.update_timestep) % 2
                        # if update_agent_num == 0:
                        #     self.env.set_prior(self.atk_prior)
                        # else:
                        #     self.env.set_prior(self.env_prior)

                        if substep != 0:
                            if update_agent_num == 0:
                                states, _, _ = self.env.sub_reset(substep, self.atk_prior)
                            else:
                                states, _, _ = self.env.sub_reset(substep, self.beliefs[substep][b])
                        else:
                            states, _, _ = self.env.reset()

                        curreward = torch.zeros((self.env.n_agents))
                        type_ob = np.zeros((1, self.env.n_targets))
                        type_ob[0, self.env.atk_type] = 1.
                        type_ob = torch.from_numpy(type_ob).float().to(device)

                        for t in range(substep, self.env.n_steps + 10):
                            # timestep += 1
                            curstep = substep
                            
                            actions = []

                            start_num = -1
                            if t == substep:
                                start_num = b

                            # for i in range(self.env.n_agents):
                            action, _ = self.agents[0].act(curstep, states[0], start_num, self.env.atk_type, in_training=True)
                            actions.append(action)
                            action, _ = self.agents[1].act(curstep, states[1], start_num, in_training=True)
                            actions.append(action)
                            v = []
                            if t != substep:
                                # for i in range(self.env.n_agents):
                                # print('state:')
                                # print(states[0][1:], np.array([states[0][1:]]))
                                v.append(self.agents[0].evaluate(t, torch.stack([states[0][1:]]), actions[0], self.env.atk_type, start_num, in_training=True)[1])
                                v.append(self.agents[1].evaluate(t, torch.stack([states[1][1:]]), actions[1], start_num, in_training=True)[1])

                            # print('actions')
                            # print(actions)
                            
                            atk_prob = torch.stack([self.agents[0].evaluate(t, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, start_num, in_training=True)[1] for tar in range(self.env.n_targets)])
                            # with torch.no_grad():
                            states, reward, done, _ = env.step(actions, atk_prob)

                            if not done and t != substep:
                                done = True
                                # print('v:')
                                # print(v)
                                reward = v
                            else:
                                reward = torch.Tensor(reward)
                            
                            self.agents[0].agents[self.env.atk_type].rewards.append(reward[0])
                            self.agents[1].rewards.append(reward[1])
                            
                            # Saving reward and is_terminal:
                            # typeob is a tensor with shape[1, :, :], sue type_ob to extract the only episode in the type_ob
                            # self.def_memory.rewards.append(reward[1])
                            # self.def_memory.is_terminals.append(done)
                            # self.def_memory.type_obs.append(type_ob[0])

                            # self.atk_memorys[self.env.atk_type].rewards.append(reward[0])
                            # self.atk_memorys[self.env.atk_type].is_terminals.append(done)
                            # self.atk_memorys[self.env.atk_type].type_obs.append(type_ob[0])
                            
                            
                            # update if its time
                            # if timestep % update_timestep == 0:
                            #     for i in range(self.env.n_agents):
                            #         agents[i][substep].update(memory[i][substep])
                            #         memorys[i][substep].clear_memory()
                            #     timestep = 0
                            
                            if substep == 0:
                                # running_reward[env.type] += reward
                                curreward += reward

                            if done:
                                # done_cnt += 1
                                # if done_cnt % self.update_timestep == 0 or i_episode == round_each_belief - 1:
                                    # for i in range(self.env.n_agents):
                                    # print('done cnt:')
                                    # print(done_cnt)
                                self.agents[0].update(substep, self.env.atk_type, self.gamma)
                                self.agents[1].update(substep, self.gamma)

                                    # if update_agent_num == 1:
                                    #     v_loss, tot_loss = self.agents[1].update(substep, self.gamma)
                                    # else:
                                    #     cntupd = 0
                                    #     v_loss, tot_loss = 0, 0
                                    #     for type_i in range(len(self.atk_memorys)):
                                    #         if len(self.atk_memorys[type_i].rewards) > 1:
                                    #             v_loss_t, tot_loss_t = self.agents[0].update(substep, self.gamma)
                                    #             cntupd += 1
                                    #             v_loss += v_loss_t
                                    #             tot_loss += tot_loss_t
                                    #         v_loss /= cntupd
                                    #         tot_loss /= cntupd
                                    # v_loss += v_loss_t
                                    # tot_loss += tot_loss_t
                                    

                                    # self.def_memory.clear_memory()
                                    
                                    # self.memorys[update_agent_num].clear_memory()
                                    # for agent_i in range(self.env.n_agents):
                                        # self.memorys[agent_i].clear_memory()
                                    
                                    # for type_i in range(len(self.atk_memorys)):
                                    #     self.atk_memorys[type_i].clear_memory()
                                    # print('episode {}: updated with v_loss {} and loss{}'.format(i_episode, v_loss, tot_loss))
                                    # update_agent_num = (update_agent_num + 1) % 2
                                    # done_cnt = 0
                                # timestep = 0
                                break

                        if i_episode % self.log_interval == 0:
                            print('{} episode trained fininshed.'.format(i_episode))

                        # Because we do not sample a full trajectory during training now, we cannot provide any information during training anymore

                        # if substep == 0:  
                        #     avg_length[env.type] += t - substep + 1
                        #     epi_cnt += 1
                        #     epi_type_cnt[env.type] += 1
                        #     running_reward[env.type] += curreward
                        
                        
                        # # logging
                        # if substep == 0 and epi_cnt % self.log_interval == 0:
                        #     running_reward = torch.zeros((self.env.n_targets, self.env.n_agents))
                        #     avg_length = np.zeros((self.env.n_targets))
                        #     epi_cnt = 0
                        #     epi_type_cnt = np.zeros((self.env.n_targets))