
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
from agent.npa_torch import NPAAgent, Memory, AtkNPAAgent
# from base_controller import BaseController
from env.base_env import BaseEnv
from datetime import datetime
from agent.npa_torch import device

import subprocess
import math
# import pickle5 as pickle
import pickle

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def one_hot(n, i):
    x = np.zeros(n)
    if i < n:
        x[i] = 1.
    return x

def save_model(model, f_path):
    with open(f_path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    # print('model saved to {}'.format(f_path))

def load_model(f_path):
    with open(f_path, 'rb') as f:
        model = pickle.load(f)
    return model

class NaiveController():
    def __init__(self, env: BaseEnv, max_episodes, lr, betas, gamma, clip_eps, n_steps, network_width, test_every, n_belief, batch_size, minibatch, k_epochs=1000, max_process=3, v_batch_size=100000, total_process=64,seed=None, n_sampler = 5, n_avrg_p=10):
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
        # self.minibatch = minibatch
        self.minibatch = batch_size # pg theorem not originally satisfied so on-policy update is needed
        self.lr = lr
        self.betas = betas
        self.gamma = gamma                # discount factor
        self.K_epochs = k_epochs                # update policy for K epochs
        self.eps_clip = clip_eps              # clip parameter for PPO
        self.random_seed = seed
        self.n_belief = n_belief
        self.max_process = max_process
        self.total_process = total_process
        self.n_sampler = n_sampler # some string process is implemented in a naive way, it will go wrong when there are more samplers than 10
        self.thread_each_process = self.total_process // (self.max_process * self.n_sampler)
        self.n_avrg_p = n_avrg_p
        
        #############################################

        torch.set_num_threads(self.thread_each_process)
        
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
                ppo = NPAAgent(self.env.n_steps, self.n_belief, self.beliefs, state_dim, action_dim, self.n_latent_var, lr, betas[i], gamma, self.K_epochs, self.eps_clip, self.minibatch, self.env.n_targets, max_n_hist=self.n_avrg_p)
            else:
                ppo = AtkNPAAgent(self.env.n_types, self.env.n_steps, self.n_belief, self.beliefs, state_dim, action_dim, self.n_latent_var, lr, betas[i], gamma, self.K_epochs, self.eps_clip, self.minibatch, self.env.n_targets, max_n_hist=self.n_avrg_p)
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

                for b in range(self.n_belief):
                    update_agent_num = 0
                    done_cnt = 0
                    for i_episode in range(round_each_belief):

                        # update_agent_num = int(i_episode / self.update_timestep) % 2
                        # if update_agent_num == 0:
                        #     self.env.set_prior(self.atk_prior)
                        # else:
                        #     self.env.set_prior(self.env_prior)

                        # if substep != 0:
                        if True:
                            if update_agent_num == 0:
                                states, _, _ = self.env.sub_reset(substep, self.atk_prior)
                            else:
                                states, _, _ = self.env.sub_reset(substep, self.beliefs[substep][b])
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

                            start_num = -1
                            if t == substep:
                                start_num = b

                            # for i in range(self.env.n_agents):
                            action, _ = self.ppos[0].act(curstep, states[0], self.atk_memorys[self.env.atk_type], start_num, self.env.atk_type, in_training=True)
                            actions.append(action)
                            action, _ = self.ppos[1].act(curstep, states[1], self.def_memory, start_num, in_training=True)
                            actions.append(action)
                            v = []
                            if t != substep:
                                # for i in range(self.env.n_agents):
                                # print('state:')
                                # print(states[0][1:], np.array([states[0][1:]]))
                                v.append(self.ppos[0].evaluate(t, torch.stack([states[0][1:]]), actions[0], self.env.atk_type, type_ob, start_num, in_training=True)[1])
                                v.append(self.ppos[1].evaluate(t, torch.stack([states[1][1:]]), actions[1], type_ob, start_num, in_training=True)[1])

                            # print('actions')
                            # print(actions)
                            
                            atk_prob = torch.stack([self.ppos[0].evaluate(t, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, np.array([one_hot(self.env.n_targets, tar)]), start_num, in_training=True)[3] for tar in range(self.env.n_targets)])
                            # with torch.no_grad():
                            states, reward, done, _ = self.env.step(actions, atk_prob)

                            if not done and t != substep:
                                done = True
                                # print('v:')
                                # print(v)
                                reward = torch.Tensor(v)
                            else:
                                reward = torch.Tensor(reward)
                            
                            # Saving reward and is_terminal:
                            # typeob is a tensor with shape[1, :, :], sue type_ob to extract the only episode in the type_ob
                            self.def_memory.rewards.append(reward[1])
                            self.def_memory.is_terminals.append(done)
                            self.def_memory.type_obs.append(type_ob[0])

                            self.atk_memorys[self.env.atk_type].rewards.append(reward[0])
                            self.atk_memorys[self.env.atk_type].is_terminals.append(done)
                            self.atk_memorys[self.env.atk_type].type_obs.append(type_ob[0])
                            
                            
                            # update if its time
                            # if timestep % update_timestep == 0:
                            #     for i in range(self.env.n_agents):
                            #         ppos[i][substep].update(memory[i][substep])
                            #         memorys[i][substep].clear_memory()
                            #     timestep = 0
                            
                            # if substep == 0:
                            #     # running_reward[env.type] += reward
                            #     curreward += reward

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
                                    
                                    # self.memorys[update_agent_num].clear_memory()
                                    # for agent_i in range(self.env.n_agents):
                                        # self.memorys[agent_i].clear_memory()
                                    
                                    for type_i in range(len(self.atk_memorys)):
                                        self.atk_memorys[type_i].clear_memory()
                                    print('{}: updated. loss: {:.4f}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), tot_loss))
                                    update_agent_num = (update_agent_num + 1) % 2
                                    # done_cnt = 0
                                # timestep = 0
                                break

                        # if substep == 0:  
                        #     avg_length[self.env.type] += t - substep + 1
                        #     epi_cnt += 1
                        #     epi_type_cnt[self.env.type] += 1
                        #     # print('type: {}, step: {}, rew: {}'.format(env.type, t, curreward))
                        #     # print(env.type)
                        #     # print('step: ')
                        #     # print(t)
                        #     # print('rew:')
                        #     # print(curreward)
                        #     running_reward[self.env.type] += curreward
                        
                        
                        # logging
                        # if substep == 0 and epi_cnt % self.log_interval == 0:
                        #     # print(avg_length, epi_cnt)
                        #     # avg_length = int(avg_length/epi_cnt)
                        #     # running_reward /= self.log_interval
                        #     # running_reward = int((running_reward/self.log_interval))

                        #     # print(epi_type_cnt)
                        #     # print('Episode {} \t episode length: {} \t reward:'.format(self.step, avg_length))
                        #     # print(running_reward)
                        #     running_reward = torch.zeros((self.env.n_targets, self.env.n_agents))
                        #     avg_length = np.zeros((self.env.n_targets))
                        #     epi_cnt = 0
                        #     epi_type_cnt = np.zeros((self.env.n_targets))

                    for i_episode in range(2 * self.v_update_timestep):
                        if True:
                            if update_agent_num == 0:
                                states, _, _ = self.env.sub_reset(substep, self.atk_prior)
                            else:
                                states, _, _ = self.env.sub_reset(substep, self.beliefs[substep][b])
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

                            start_num = -1
                            if t == substep:
                                start_num = b

                            action, _ = self.ppos[0].act(curstep, states[0], self.atk_memorys[self.env.atk_type], start_num, self.env.atk_type, in_training=True)
                            actions.append(action)
                            action, _ = self.ppos[1].act(curstep, states[1], self.def_memory, start_num, in_training=True)
                            actions.append(action)
                            v = []
                            if t != substep:
                                v.append(self.ppos[0].evaluate(t, torch.stack([states[0][1:]]), actions[0], self.env.atk_type, type_ob, start_num, in_training=False)[1])
                                v.append(self.ppos[1].evaluate(t, torch.stack([states[1][1:]]), actions[1], type_ob, start_num, in_training=False)[1])
                                done = True
                                reward = torch.Tensor(v)
                            else:

                                atk_prob = torch.stack([self.ppos[0].evaluate(t, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, np.array([one_hot(self.env.n_targets, tar)]), start_num, in_training=True)[3] for tar in range(self.env.n_targets)])
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
            save_model(atk_dict, 'models/atk_model_r_{}_bsize_{}_time_{}.pickle'.format(round_each_belief, self.update_timestep, datetime.now().strftime("%m:%d,%H:%M:%S")))
            save_model(def_dict, 'models/def_model_r_{}_bsize_{}_time_{}.pickle'.format(round_each_belief, self.update_timestep, datetime.now().strftime("%m:%d,%H:%M:%S")))
        
    def train_main_process(self, num_round=None, round_each_belief = 1000, policy_store_every=100, 
               sec_prob=False, save_every=None, save_path=None, load_state=None, load_path=None, store_results=False, subpros_name='run_tag_subp', continue_name=None):
        self.step = 0
        env = self.env
        # if num_round == None:
            # num_round = self.max_episodes
        
        running_reward = torch.zeros((self.env.n_targets, self.env.n_agents))
        avg_length = np.zeros((self.env.n_targets))
        timestep = 0
        epi_cnt = 0
        epi_type_cnt = np.zeros((self.env.n_targets))
        done_cnt = 0

        exp_name = continue_name if continue_name != None else 'model_r_{}_bsize_{}_time_{}'.format(round_each_belief, self.update_timestep, datetime.now().strftime("%m:%d,%H:%M:%S"))

        def train_and_fetch_subprocess(substep, b_st, b_ed):
            sp_lists = []
            all_done = False
            while not all_done:
                for b in range(b_st, b_ed):
                    if os.path.exists('models/atk_{}_round_{}_belief_{}.pickle'.format(exp_name, substep, b)) and os.path.exists('models/def_{}_round_{}_belief_{}.pickle'.format(exp_name, substep, b)):
                        continue

                    arg = ["python", "{}.py".format(subpros_name), "--n-belief={}".format(self.n_belief), "--n-steps={:d}".format(self.env.n_steps), "--learning-rate={}".format(self.lr), "--exp-name={}".format(exp_name), "--train-round={}".format(substep), "--train-belief={}".format(b), "--batch-size={}".format(self.update_timestep), "--minibatch={}".format(self.minibatch), "--max-steps={}".format(round_each_belief), "--seed={}".format(self.random_seed), "--k-epochs={}".format(self.K_epochs), "--v-batch-size={}".format(self.v_update_timestep), "--num-thread={}".format(self.thread_each_process), "--n-sampler={}".format(self.n_sampler), "--n-avrg-p={}".format(self.n_avrg_p)]

                    sp = subprocess.Popen(arg)
                    sp_lists.append(sp)
                    
                for sp in sp_lists:
                    sp.wait()
                
                this_time_finish = True
                    
                for b in range(b_st, b_ed):
                    try:
                        atk_dict_path = 'models/atk_{}_round_{}_belief_{}.pickle'.format(exp_name, substep, b)
                        atk_dict_to_load = load_model(atk_dict_path)
                        def_dict_path = 'models/def_{}_round_{}_belief_{}.pickle'.format(exp_name, substep, b)
                        def_dict_to_load = load_model(def_dict_path)
                        self.ppos[0].load_model_with_specify(substep, b, atk_dict_to_load)
                        self.ppos[1].load_model_with_specify(substep, b, def_dict_to_load)
                    except:
                        this_time_finish = False
                        print('round {}, belief {} not finish correctly!'.format(substep, b))
                        break

                all_done = this_time_finish
                # self.agent

        # while self.step < num_round:
            # self.step += 1
        # for substep in range(self.env.n_steps - 1, -1, -1):
        if True:
            substep = num_round
            print('start training substep {}.'.format(substep))

            if substep != self.env.n_steps - 1:
                self.load_all_models(exp_name)
            sp_lists = []
            # for b in range(self.n_belief):
            for b_batch in range(math.ceil(1. * self.n_belief / self.max_process)):
                b_st = b_batch * self.max_process
                b_ed = min((b_batch + 1) * (self.max_process), self.n_belief)
                train_and_fetch_subprocess(substep, b_st, b_ed)

            atk_dict = self.ppos[0].get_state_dict()
            def_dict = self.ppos[1].get_state_dict()

            save_model(atk_dict, 'models/atk_{}.pickle'.format(exp_name))
            save_model(def_dict, 'models/def_{}.pickle'.format(exp_name))   
                # for i in range():

                
    def load_all_models(self, exp_name):
        atk_dict_path = 'models/atk_{}.pickle'.format(exp_name)
        atk_dict_to_load = load_model(atk_dict_path)
        def_dict_path = 'models/def_{}.pickle'.format(exp_name)
        def_dict_to_load = load_model(def_dict_path)

        # print(len(atk_dict_to_load))
        # print(len(atk_dict_to_load[0]))

        self.ppos[0].set_state_dict(atk_dict_to_load)
        self.ppos[1].set_state_dict(def_dict_to_load)

    def test_env_input(self, exp_name):
        self.load_all_models(exp_name)

        print('input round, zero base')
        cur_round = int(input())
        print('input atk place')
        atk_x, atk_y = input().split()
        atk_x = int(atk_x)
        atk_y = int(atk_y)
        atk_p = np.array([atk_x, atk_y])
        print('input def place')
        def_x, def_y = input().split()
        def_x = int(def_x)
        def_y = int(def_y)
        def_p = np.array([def_x, def_y])
        print('input belief')
        b1, b2 = input().split()
        b1 = float(b1)
        b2 = float(b2)
        b = torch.Tensor([b1, b2])

        states, _, _ = self.env.reset_to_state_with_type((atk_p, def_p), b, 0)

        print('test state is:')
        print(states)

        print('atk_strategy:')
        # print(atk_strategy)
        for type_i in range(self.env.n_types):
            _, atk_t_strategy = self.ppos[0].act(cur_round, self._get_atk_ob_full(type_i, self.env.belief, states[0])[0], self.ppos[0].memory_ph, type_n=type_i)
            print(atk_t_strategy)
        
        atk_action = 0
        def_action, def_strategy = self.ppos[1].act(cur_round, states[1], self.ppos[1].memory_ph)
        # states, rew, done, _ = self.env.step(actions, atk_prob, verbose=True)
        
        print('def_strategy:')
        print(def_strategy)
        print('value:')
        atk_values = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), atk_action, tar, np.array([one_hot(self.env.n_targets, tar)]))[1].detach() for tar in range(self.env.n_targets)])
        def_values = self.ppos[1].evaluate(cur_round, torch.stack([states[1][1:]]), def_action, np.array([one_hot(self.env.n_targets, self.env.atk_type)]))[1].detach()
        print(atk_values)
        print(def_values)

    # sample the samples for train_belief
    def sample_belief(self, substep, b, n_samples, exp_name, update_agent_num):
        # while the input exp_name has the sampler number in it, we need to remove it to get the current model
        true_exp_name = exp_name[:-10]
        # print('true_exp_name:')
        # print(true_exp_name)

        # load all the models to be sampled
        if substep != self.env.n_steps - 1:
            self.load_all_models(true_exp_name)
        
        atk_dict_path = 'models/atk_{}_round_{}_belief_{}.pickle'.format(true_exp_name, substep, b)
        atk_dict_to_load = load_model(atk_dict_path)
        def_dict_path = 'models/def_{}_round_{}_belief_{}.pickle'.format(true_exp_name, substep, b)
        def_dict_to_load = load_model(def_dict_path)
        self.ppos[0].load_model_with_specify(substep, b, atk_dict_to_load)
        self.ppos[1].load_model_with_specify(substep, b, def_dict_to_load)

        # 

        # update_agent_num = 0
        done_cnt = 0

        all_rewards_in_train = []

        for i_episode in range(self.update_timestep):
            # print('policy {} updating'.format(i_episode))

            # update_agent_num = int(i_episode / self.update_timestep) % 2
            # if update_agent_num == 0:
            #     self.env.set_prior(self.atk_prior)
            # else:
            #     self.env.set_prior(self.env_prior)

            # if substep != 0:
            if True:
                if update_agent_num == 0:
                    states, _, _ = self.env.sub_reset(substep, self.atk_prior)
                else:
                    # if b == 0:
                        # print('now updating defender', self.beliefs[substep][b])
                    states, _, _ = self.env.sub_reset(substep, self.beliefs[substep][b])
            else:
                states, _, _ = self.env.reset()

            if b == 0 and update_agent_num == 1 and self.env.atk_type == 0:
                print(self.beliefs[substep][b], self.env.atk_type)
            # curreward = torch.zeros((self.env.n_agents))
            type_ob = np.zeros((1, self.env.n_targets))
            type_ob[0, self.env.atk_type] = 1.
            type_ob = torch.from_numpy(type_ob).float().to(device)

            cur_rew = torch.Tensor([0, 0])

            for t in range(substep, self.env.n_steps + 10):
                # timestep += 1
                curstep = substep
                
                actions = []

                start_num = -1
                if t == substep:
                    start_num = b

                # for i in range(self.env.n_agents):
                action, _ = self.ppos[0].act(curstep, states[0], self.atk_memorys[self.env.atk_type], start_num, self.env.atk_type, in_training=True)
                actions.append(action)
                action, _ = self.ppos[1].act(curstep, states[1], self.def_memory, start_num, in_training=True)
                actions.append(action)
                v = []
                if t != substep:
                    # for i in range(self.env.n_agents):
                    # print('not first step, state:')
                    # print(states[0][1:], np.array([states[0][1:]]))
                    v.append(self.ppos[0].evaluate(t, torch.stack([states[0][1:]]), actions[0], self.env.atk_type, type_ob, start_num, in_training=False)[1])
                    v.append(self.ppos[1].evaluate(t, torch.stack([states[1][1:]]), actions[1], type_ob, start_num, in_training=False)[1])
                    done = True
                    reward = torch.Tensor(v)

                    # self.def_memory.states.append(observation[1 + self.n_target:])
                    # self.def_memory.actions.append(action.detach())
                    # self.def_memory.logprobs.append(dist.log_prob(action).detach())

                    # print(reward)
                else:

                    # print('actions')
                    # print(actions)
                    
                    atk_prob = torch.stack([self.ppos[0].evaluate(t, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, np.array([one_hot(self.env.n_targets, tar)]), start_num, in_training=True)[3] for tar in range(self.env.n_targets)])
                    # with torch.no_grad():
                    states, reward, done, _ = self.env.step(actions, atk_prob)
                    reward = torch.Tensor(reward)
                    # print('first step, done:')
                    # print(done)
                    # print(done, type(reward[0]))
                
                # Saving reward and is_terminal:
                # typeob is a tensor with shape[1, :, :], sue type_ob to extract the only episode in the type_ob
                self.def_memory.rewards.append(reward[1])
                self.def_memory.is_terminals.append(done)
                self.def_memory.type_obs.append(type_ob[0])

                self.atk_memorys[self.env.atk_type].rewards.append(reward[0])
                self.atk_memorys[self.env.atk_type].is_terminals.append(done)
                self.atk_memorys[self.env.atk_type].type_obs.append(type_ob[0])

                cur_rew += reward

                if done:
                    break
        
        if update_agent_num == 1:
            # tot_loss = self.ppos[1].update(substep, self.def_memory)
            self.ppos[1].update(substep, self.def_memory)
            def_all_grads = self.ppos[1].get_all_grads(substep, b)
            save_model(def_all_grads, 'results/grad_round_{}_belief_{}_{}.pickle'.format(substep, b, exp_name))
        else:
            cntupd = 0
            v_loss, tot_loss = 0, 0
            for type_i in range(len(self.atk_memorys)):
                if len(self.atk_memorys[type_i].rewards) > 1:
                    # tot_loss_t = self.ppos[0].update(substep, self.atk_memorys[type_i], type_i)
                    self.ppos[0].update(substep, self.atk_memorys[type_i], type_i)
                    # cntupd += 1
                    # v_loss += v_loss_t
                    # tot_loss += tot_loss_t
            # v_loss /= cntupd
            # tot_loss /= cntupd
            atk_all_grads = self.ppos[0].get_all_grads(substep, b)
            save_model(atk_all_grads, 'results/grad_round_{}_belief_{}_{}.pickle'.format(substep, b, exp_name))
                
        # sampler_name = exp_name + '_sampler'
        # self.def_memory.save_samples('results/def_memory_round_{}_belief_{}_{}.pickle'.format(substep, b, exp_name))
        # for type_i in range(self.env.n_types):
        #     self.atk_memorys[type_i].save_samples('results/atk_memory_type_{}_round_{}_belief_{}_{}.pickle'.format(type_i, substep, b, exp_name))
        
                        

    # train the <b> th belief of round <substep> for <round_each_belief> time
    def train_belief(self, substep, b, round_each_belief, exp_name, subpros_name='run_tag_sample'):
        if substep != self.env.n_steps - 1:
            self.load_all_models(exp_name)
        # else:
            # round_each_belief = 0

        update_agent_num = 0
        done_cnt = 0

        all_rewards_in_train = []

        atk_dict = self.ppos[0].get_state_dict()
        def_dict = self.ppos[1].get_state_dict()

        starting_time = datetime.now()

        save_model(atk_dict, 'models/atk_{}_round_{}_belief_{}.pickle'.format( exp_name, substep, b))
        save_model(def_dict, 'models/def_{}_round_{}_belief_{}.pickle'.format( exp_name, substep, b))
        for i_episode in range(round_each_belief // self.update_timestep):

            # for num_trained in range()
            sp_lists = []
            for i_samplers in range(self.n_sampler):
                # print()
                arg = ["python", "{}.py".format(subpros_name), "--n-belief={}".format(self.n_belief), "--n-steps={:d}".format(self.env.n_steps), "--learning-rate={}".format(self.lr), "--exp-name={}_sampler_{}".format(exp_name, i_samplers), "--train-round={}".format(substep), "--train-belief={}".format(b), "--batch-size={}".format(self.update_timestep), "--minibatch={}".format(self.minibatch), "--max-steps={}".format(self.update_timestep), "--seed={}".format(self.random_seed + i_samplers), "--k-epochs={}".format(self.K_epochs), "--v-batch-size={}".format(self.v_update_timestep), "--num-thread={}".format(self.thread_each_process), "--update-agent-num={}".format(update_agent_num), "--n-sampler={}".format(self.n_sampler), "--n-avrg-p={}".format(self.n_avrg_p)]

                sp = subprocess.Popen(arg)
                sp_lists.append(sp)

            for sp in sp_lists:
                sp.wait()
            
            self.ppos[0].all_zero_grad()
            self.ppos[1].all_zero_grad()
            for i_samplers in range(self.n_sampler):
                sampler_name = '{}_sampler_{}'.format(exp_name, i_samplers)
                # for type_i in range(self.env.n_types):
                    # self.atk_memorys[type_i].load_samples('results/atk_memory_type_{}_round_{}_belief_{}_{}.pickle'.format(type_i, substep, b, sampler_name))
                if update_agent_num == 1: 
                    def_grad_d_sampler_i = load_model('results/grad_round_{}_belief_{}_{}.pickle'.format(substep, b, sampler_name))
                    self.ppos[1].load_all_grads(def_grad_d_sampler_i, substep, b)
                    # print('defender:', def_grad_d_sampler_i)
                else:
                    # print(i_samplers, 'load grad finish!')
                    atk_grad_d_sampler_i = load_model('results/grad_round_{}_belief_{}_{}.pickle'.format(substep, b, sampler_name))
                    self.ppos[0].load_all_grads(atk_grad_d_sampler_i, substep, b)
                    # print('atk:', atk_grad_d_sampler_i)
                # print(i_samplers, 'load grad finish!')
                # self.def_memory.load_samples('results/def_memory_round_{}_belief_{}_{}.pickle'.format(substep, b, sampler_name))
            self.ppos[0].do_optimize(substep, b)
            self.ppos[1].do_optimize(substep, b)

            # print('{}: episode {} start training.'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), i_episode), end = ' ')
                        
            
            # v_loss += v_loss_t
            # tot_loss += tot_loss_t
            save_model(atk_dict, 'models/atk_{}_round_{}_belief_{}.pickle'.format( exp_name, substep, b))
            save_model(def_dict, 'models/def_{}_round_{}_belief_{}.pickle'.format( exp_name, substep, b))

            # self.def_memory.clear_memory()
                        
            # for type_i in range(len(self.atk_memorys)):
            #     self.atk_memorys[type_i].clear_memory()

            # self.def_memory.clear_memory()
            
            # self.memorys[update_agent_num].clear_memory()
            # for agent_i in range(self.env.n_agents):
            #     self.memorys[agent_i].clear_memory()
            
            # for type_i in range(len(self.atk_memorys)):
            #     self.atk_memorys[type_i].clear_memory()
            # print('{}: updated. loss: {:.4f}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), tot_loss))
            expected_finish_time = starting_time + (datetime.now() - starting_time) * (round_each_belief // self.update_timestep) / (i_episode + 1)
            print('Trained {}\%. Expected policy training finish time: {}'.format((i_episode + 1) *100 / (round_each_belief // self.update_timestep), expected_finish_time.strftime("%m/%d/%Y, %H:%M:%S")))
            update_agent_num = (update_agent_num + 1) % 2

            # save_model()
            

            # done_cnt = 0
        # timestep = 0
        starting_time = datetime.now()
        
        # print('tot v step:')
        # print(self.v_update_timestep)

        for i_episode in range(2 * self.v_update_timestep):
            if True:
                if update_agent_num == 0:
                    states, _, _ = self.env.sub_reset(substep, self.atk_prior)
                else:
                    states, _, _ = self.env.sub_reset(substep, self.beliefs[substep][b])
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

                start_num = -1
                if t == substep:
                    start_num = b

                action, _ = self.ppos[0].act(curstep, states[0].detach(), self.atk_memorys[self.env.atk_type], start_num, self.env.atk_type, in_training=(t == substep))
                actions.append(action)
                action, _ = self.ppos[1].act(curstep, states[1].detach(), self.def_memory, start_num, in_training=(t == substep))
                actions.append(action)
                v = []
                if t != substep:
                    v.append(self.ppos[0].evaluate(t, torch.stack([states[0][1:]]), actions[0], self.env.atk_type, type_ob, start_num, in_training=False)[1].detach())
                    v.append(self.ppos[1].evaluate(t, torch.stack([states[1][1:]]), actions[1], type_ob, start_num, in_training=False)[1].detach())
                    done=True
                    reward = torch.Tensor(v)
                else:
                    atk_prob = torch.stack([self.ppos[0].evaluate(t, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, np.array([one_hot(self.env.n_targets, tar)]), start_num, in_training=True)[3].detach() for tar in range(self.env.n_targets)])
                    states, reward, done, _ = self.env.step(actions, atk_prob)
                    # states = states.detach()
                    # states = [states[0]]
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
                    if done_cnt % self.v_update_timestep == 0:
                        print('{}: episode {} start training.'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), i_episode), end = ' ')
                        
                        if update_agent_num == 1:
                            v_loss = self.ppos[1].v_update(substep, self.def_memory)
                        else:
                            cntupd = 0
                            v_loss, tot_loss = 0, 0
                            for type_i in range(len(self.atk_memorys)):
                                if len(self.atk_memorys[type_i].rewards) > 1:
                                    # print('atk type {} trained'.format(type_i))

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
        # print(len(atk_dict))
        # print(len(atk_dict[0]))

        def_dict = self.ppos[1].get_state_dict()
        # print(os.path.exists('models'))
        save_model(atk_dict, 'models/atk_{}_round_{}_belief_{}.pickle'.format( exp_name, substep, b))
        save_model(def_dict, 'models/def_{}_round_{}_belief_{}.pickle'.format( exp_name, substep, b))
        # save_model(all_rewards_in_train, 'results/reward_{}_round_{}_belief_{}.pickle'.format(exp_name, substep, b))
    
    def sub_game_exploitability(self, single_train_round=20000, episodes_test=100, exp_name=None):
        if exp_name != None:
            self.load_all_models(exp_name)

        print('model load finish')
        
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
                atk_action, _ = self.ppos[0].act(current_len - 1, states[0], self.atk_memorys[self.env.atk_type], -1, type_n=self.env.atk_type)
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
                atk_prob = torch.stack([self.ppos[0].evaluate(current_len - 1, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, np.array([one_hot(self.env.n_targets, tar)]), -1, in_training=False)[3] for tar in range(self.env.n_targets)])
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
                # print('after step:')
                # print(current_len, done)
                if current_len > 1:
                    train_memory.next_vs.append(v)
                    # self.atk_memorys[self.env.atk_type].next_vs.append(v[0])
                if done:
                    train_memory.next_vs.append(torch.zeros_like(v))
                    # print('done!')
                    # print(self.step, self.update_timestep)
                    # self.atk_memorys[self.env.atk_type].next_vs.append(torch.zeros_like(v[0]))

                # update if its time
                if done and self.step % self.update_timestep == 0:
                    print('updated')
                    print(self.step, single_train_round)
                    # train_agent_n = int(self.step / self.update_timestep) % 2
                    # for i in range(self.env.n_agents):
                    # v_loss = self.ppos[train_agent_n].update(self.memorys[train_agent_n])
                    v_loss = 0
                    train_agent.update(train_memory)
                    # for agent_i in range(self.env.n_agents):
                        # self.memorys[agent_i].clear_memory()
                    for type_i in range(self.env.n_types):
                        self.atk_memorys[type_i].clear_memory()
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
                                    states, _, _ = self.env.reset_to_state((np.array([0, 4]), np.array([1, 4])), torch.Tensor(self.env.prior))
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
                                    if def_strategies[def_a_3] < 1e-6:
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
                                                type_possis = self.env.get_current_state()[1]
                                        else:
                                            atk_action, atk_strategy = self.ppos[0].act(cur_round, states[0], self.ppos[0].memory_ph, type_n=cur_type)
                                            def_action, def_strategy = train_agent.act(cur_round, states[1], train_memory)
                                            # atk_prob = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), atk_action, tar, np.array([one_hot(self.env.n_targets, tar)]))[1].detach() for tar in range(self.env.n_targets)])
                                            actions = [atk_action, def_action]
                                        
                                        atk_prob = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, np.array([one_hot(self.env.n_targets, tar)]), -1, in_training=False)[3] for tar in range(self.env.n_targets)])
                                        
                                        states, rew, done, _ = self.env.step(actions, atk_prob, verbose=True)

                                        # if not is_second_round and is_third_round and type(def_strategies) == NoneType:
                                            # def_strategies = self.env.

                                        if need_get_type:
                                            poss, belief = self.env.get_current_state()
                                            states, _, _ = self.env.reset_to_state(poss, belief)
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
                                        cur_def_rew += def_strategies[def_a_3] * def_rews[type_i] / type_cnt[type_i]
                                        cur_atk_rew += def_strategies[def_a_3] * atk_rews[type_i] / type_cnt[type_i]
                            for type_i in range(self.env.n_types):
                                if cur_atk_rew[type_i] > cur_max_atk[type_i]:
                                    cur_max_atk[type_i] = cur_atk_rew[type_i]
                                    cur_max_def[type_i] = cur_def_rew[type_i]    

                        # full_defs.append((def_rews[0] + def_rews[1]) * 1./ (type_cnt[0] + type_cnt[1]))
                        full_defs.append(def_rew[0] * type_possis[0] + def_rew[1] * type_possis[1])
                        full_atks[0].append(cur_max_atk[0])
                        full_atks[1].append(cur_max_atk[1])
                        # if type_cnt[0] > 0:
                        #     full_atks[0].append(atk_rews[0] / type_cnt[0])
                        # else:
                        #     full_atks[0].append(0.)
                        # if type_cnt[1] > 0:
                        #     full_atks[1].append(atk_rews[1] / type_cnt[1])
                        # else:
                        #     full_atks[1].append(0.)
        print(full_defs)
        print(full_atks)
                        # for type_i in range(self.env.n_types):
                        #     if type_cnt[type_i] != 0:
                        #         print('type {}: sampled {} times, atk_avg_rew: {}, def_avg_rew: {}'.format(type_i, type_cnt[type_i], atk_rews[type_i] / type_cnt[type_i], def_rews[type_i] / type_cnt[type_i]))
    
    def calculate_exploitability(self, single_train_round=100000, episodes_test=100, exp_name=None):
        if exp_name != None:
            self.load_all_models(exp_name)

        print('model load finish')
        
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
                atk_action, _ = self.ppos[0].act(current_len - 1, states[0], self.atk_memorys[self.env.atk_type], -1, type_n=self.env.atk_type)
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
                atk_prob = torch.stack([self.ppos[0].evaluate(current_len - 1, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, np.array([one_hot(self.env.n_targets, tar)]), -1, in_training=False)[3] for tar in range(self.env.n_targets)])
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
                # print('after step:')
                # print(current_len, done)
                if current_len > 1:
                    train_memory.next_vs.append(v)
                    # self.atk_memorys[self.env.atk_type].next_vs.append(v[0])
                if done:
                    train_memory.next_vs.append(torch.zeros_like(v))
                    # print('done!')
                    # print(self.step, self.update_timestep)
                    # self.atk_memorys[self.env.atk_type].next_vs.append(torch.zeros_like(v[0]))

                # update if its time
                if done and self.step % self.update_timestep == 0:
                    print('updated')
                    print(self.step, single_train_round)
                    # train_agent_n = int(self.step / self.update_timestep) % 2
                    # for i in range(self.env.n_agents):
                    # v_loss = self.ppos[train_agent_n].update(self.memorys[train_agent_n])
                    v_loss = 0
                    train_agent.update(train_memory)
                    # for agent_i in range(self.env.n_agents):
                        # self.memorys[agent_i].clear_memory()
                    for type_i in range(self.env.n_types):
                        self.atk_memorys[type_i].clear_memory()
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
                atk_action, atk_strategy = self.ppos[0].act(cur_round, states[0], self.ppos[0].memory_ph, type_n=cur_type)
                def_action, def_strategy = train_agent.act(cur_round, states[1], train_memory)
                # atk_prob = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), atk_action, tar, np.array([one_hot(self.env.n_targets, tar)]))[1].detach() for tar in range(self.env.n_targets)])
                actions = [atk_action, def_action]
                atk_prob = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), actions[0], tar, np.array([one_hot(self.env.n_targets, tar)]), -1, in_training=False)[3] for tar in range(self.env.n_targets)])
                
                states, rew, done, _ = self.env.step(actions, atk_prob, verbose=True)
                print('atk_strategy:')
                # print(atk_strategy)
                for type_i in range(self.env.n_types):
                    _,  atk_t_strategy = self.ppos[0].act(cur_round, states[0], self.ppos[0].memory_ph, type_n=type_i)
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
            # states, _, _ = self.env.reset()
            # print('new episode:')
            states, _, _ = self.env.sub_reset(0, self.env.prior)
            cur_type = self.env.atk_type
            type_cnt[cur_type] += 1
            done = False
            atk_rew = 0
            def_rew = 0
            while not done:
                atk_action, atk_strategy = self.ppos[0].act(cur_round, states[0], self.ppos[0].memory_ph, type_n=cur_type)
                def_action, def_strategy = self.ppos[1].act(cur_round, states[1], self.ppos[1].memory_ph)
                atk_prob = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), atk_action, tar, np.array([one_hot(self.env.n_targets, tar)]))[3].detach() for tar in range(self.env.n_targets)])
                actions = [atk_action, def_action]
                print('atk_strategy:')
                # print(atk_strategy)
                for type_i in range(self.env.n_types):
                    _, atk_t_strategy = self.ppos[0].act(cur_round, self._get_atk_ob_full(type_i, self.env.belief, states[0])[0], self.ppos[0].memory_ph, type_n=type_i)
                    print(atk_t_strategy)

                states, rew, done, _ = self.env.step(actions, atk_prob, verbose=True)
                
                print('def_strategy:')
                print(def_strategy)
                print('value:')
                atk_values = torch.stack([self.ppos[0].evaluate(cur_round, self._get_atk_ob(tar, self.env.belief, states[0]), atk_action, tar, np.array([one_hot(self.env.n_targets, tar)]))[1].detach() for tar in range(self.env.n_targets)])
                def_values = self.ppos[1].evaluate(cur_round, torch.stack([states[1][1:]]), def_action, np.array([one_hot(self.env.n_targets, self.env.atk_type)]))[1].detach()
                print(atk_values)
                print(def_values)
                # print(self.ppos[0].evaluate(curround, states[0], atk_action, ))

                atk_rew += rew[0]
                def_rew += rew[1]
            atk_rews[cur_type] += atk_rew
            def_rews[cur_type] += def_rew

        for type_i in range(self.env.n_types):
            if type_cnt[type_i] != 0:
                print('type {}: sampled {} times, atk_avg_rew: {}, def_avg_rew: {}'.format(type_i, type_cnt[type_i], atk_rews[type_i] / type_cnt[type_i], def_rews[type_i] / type_cnt[type_i]))