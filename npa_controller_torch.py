
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
from agent.npa_torch import NPAAgent, Memory
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
        self.K_epochs = 10                # update policy for K epochs
        self.eps_clip = clip_eps              # clip parameter for PPO
        self.random_seed = seed
        self.n_belief = n_belief
        #############################################
        
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            # env.seed(random_seed)
        
        self.memorys = []
        self.ppos = []
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

        for i in range(env.n_agents):
            curmemory = []
            curppo = []
            # for j in range(env.n_steps):
            memory = Memory()
            state_dim = self.env.observation_spaces[i].shape[0]
            action_dim = self.env.action_spaces[i].n
            ppo = NPAAgent(self.env.n_steps, self.n_belief_n[i], self.beliefs_n[i], state_dim, action_dim, self.n_latent_var, lr, betas[i], gamma, self.K_epochs, self.eps_clip, self.minibatch, self.env.n_targets)
                # curmemory.append(memory)
                # curppo.append(ppo)
            self.memorys.append(memory)
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

    def _get_atk_ob(self, target, belief, last_obs_atk) :
        ret = np.copy(last_obs_atk)
        # print(ret.shape)
        ret = ret[1 + self.env.n_targets * 2:]
        ret = np.array([np.concatenate((one_hot(self.env.n_targets, target), belief, ret))])
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
                    update_agent_num = 0
                    done_cnt = 0
                    for i_episode in range(round_each_belief):

                        # update_agent_num = int(i_episode / self.update_timestep) % 2
                        if update_agent_num == 0:
                            self.env.set_prior(self.atk_prior)
                        else:
                            self.env.set_prior(self.env_prior)

                        if substep != 0:
                            states, _, _ = self.env.sub_reset(substep, self.beliefs[substep][b])
                        else:
                            states, _, _ = self.env.reset()
                        


                        curreward = np.zeros((self.env.n_agents))
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
                            start_nums = [self.env.atk_type, start_num]

                            # for i in range(self.env.n_agents):
                            action, _ = self.ppos[0].act(curstep, states[0], self.memorys[0], self.env.atk_type)
                            actions.append(action)
                            action, _ = self.ppos[1].act(curstep, states[1], self.memorys[1], start_num)
                            actions.append(action)
                            v = []
                            if t != substep:
                                for i in range(self.env.n_agents):
                                    v.append(self.ppos[i].evaluate(t, np.array([states[i][1:]]), actions[i], start_nums[i], type_ob)[1].cpu().detach().numpy())

                            # print('actions')
                            # print(actions)
                            
                            atk_prob = [self.ppos[0].evaluate(t, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], self.env.atk_type, type_ob)[3].cpu().detach().numpy() for tar in range(self.env.n_targets)]
                            with torch.no_grad():
                                states, reward, done, _ = env.step(actions, atk_prob)
                            


                            if not done and t != substep:
                                done = True
                                # print('v:')
                                # print(v)
                                reward = v
                            
                            # Saving reward and is_terminal:
                            for i in range(self.env.n_agents):
                                self.memorys[i].rewards.append(reward[i])
                                self.memorys[i].is_terminals.append(done)
                                self.memorys[i].type_obs.append(type_ob[0])
                            
                            # update if its time
                            # if timestep % update_timestep == 0:
                            #     for i in range(self.env.n_agents):
                            #         ppos[i][substep].update(memory[i][substep])
                            #         memorys[i][substep].clear_memory()
                            #     timestep = 0
                            
                            if substep == 0:
                                # running_reward[env.type] += reward
                                curreward += reward

                            if done:
                                done_cnt += 1
                                if done_cnt % self.update_timestep == 0 or i_episode == round_each_belief - 1:
                                    # for i in range(self.env.n_agents):
                                    # print('done cnt:')
                                    # print(done_cnt)
                                    v_loss, tot_loss = self.ppos[update_agent_num].update(substep, self.memorys[update_agent_num])
                                    # self.memorys[update_agent_num].clear_memory()
                                    for agent_i in range(self.env.n_agents):
                                        self.memorys[agent_i].clear_memory()
                                    print('episode {}: {} agent updated with v_loss {} and loss{}'.format(i_episode, update_agent_num, v_loss, tot_loss))
                                    update_agent_num = (update_agent_num + 1) % 2
                                    # done_cnt = 0
                                # timestep = 0
                                break

                        if substep == 0:  
                            avg_length[env.type] += t - substep + 1
                            epi_cnt += 1
                            epi_type_cnt[env.type] += 1
                            # print('type: {}, step: {}, rew: {}'.format(env.type, t, curreward))
                            # print(env.type)
                            # print('step: ')
                            # print(t)
                            # print('rew:')
                            # print(curreward)
                            running_reward[env.type] += curreward
                        
                        
                        # logging
                        if substep == 0 and epi_cnt % self.log_interval == 0:
                            # print(avg_length, epi_cnt)
                            # avg_length = int(avg_length/epi_cnt)
                            # running_reward /= self.log_interval
                            # running_reward = int((running_reward/self.log_interval))

                            # print(epi_type_cnt)
                            # print('Episode {} \t episode length: {} \t reward:'.format(self.step, avg_length))
                            # print(running_reward)
                            running_reward = np.zeros((self.env.n_targets, self.env.n_agents))
                            avg_length = np.zeros((self.env.n_targets))
                            epi_cnt = 0
                            epi_type_cnt = np.zeros((self.env.n_targets))