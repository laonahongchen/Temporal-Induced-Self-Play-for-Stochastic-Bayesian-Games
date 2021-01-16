import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np
import gc
import math
import pickle

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# from npa_controller_torch import device

def save_model(model, f_path):
    with open(f_path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    # print('samples saved to {}'.format(f_path))

def load_model(f_path):
    with open(f_path, 'rb') as f:
        model = pickle.load(f)
    return model

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.type_obs = []
        self.next_vs = []
        self.start_num = 0
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.type_obs[:]
        del self.next_vs[:]
        # del self.start_num
        gc.collect()
    
    def first_step(self, reward):
        ret = Memory()
        fetch = True
        for i in range(len(self.actions)):
            if fetch:
                ret.actions.append(self.actions[i])
                ret.states.append(self.states[i])
                ret.logprobs.append(self.logprobs[i])
                ret.rewards.append(self.rewards[i])
                # ret.is_terminals.append(self.is_terminals[i])
                ret.is_terminals.append(True)
                ret.type_obs.append(self.type_obs[i])
                if self.is_terminals[i]:
                    # print(type(self.rewards[i]))
                    ret.next_vs.append(torch.zeros_like(self.rewards[i]))
                else:
                    ret.next_vs.append(self.rewards[i + 1])
                ret.start_num = self.start_num
            fetch = self.is_terminals[i]
        # ret.actions = self.actions[0:1]
        # ret.states = self.states[0:1]
        # ret.logprobs = self.logprobs[0:1]
        # ret.rewards = [reward]
        # ret.is_terminals = self.is_terminals[0:1]

        return ret
    
    def load_samples(self, name):
        all_lists = load_model(name)
        self.actions += all_lists[0]
        self.states += all_lists[1]
        self.logprobs += all_lists[2]
        self.rewards += all_lists[3]
        self.is_terminals += all_lists[4]
        self.type_obs += all_lists[5]
        self.next_vs += all_lists[6]
        self.start_num = all_lists[7]

    def save_samples(self, name):
        save_lists = [self.actions, self.states, self.logprobs, self.rewards, self.is_terminals, self.type_obs, self.next_vs, self.start_num]
        save_model(save_lists, name)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, type_dim = 0):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                # nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                # nn.ReLU(),
                nn.Tanh(),
                # nn.Linear(n_latent_var, n_latent_var),
                # nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
    def act(self, state, memory):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().to(device) 
        else:
            state = state.to(device)
        action_probs = self.action_layer(state)
        
        return action_probs
    
    def evaluate(self, state, action, typeob):
        # print(type(state))
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        else:
            state = state.to(device)
        if type(typeob) != torch.Tensor:
            typeob = torch.from_numpy(typeob).float().to(device)
        else:
            typeob = typeob.to(device)
        # print('now in evaluate')
        # print(state.shape, typeob.shape)

        value_state = torch.cat([state], dim=1)
        action_probs = self.action_layer(state)

        action_prob = action_probs[:, action]

        # print('state 3:')
        # print(state)

        if type(action) != torch.Tensor:
            action = torch.tensor([action]).to(device)

        # print('eval action:')
        # print(state)
        # print(action)
        # print(action_probs)

        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)

        # print(action_logprobs)

        dist_entropy = dist.entropy()

        # state_value = self.value_layer(state)
        return action_probs, torch.Tensor(0), action_logprobs, dist_entropy # action_logprobs, torch.squeeze(state_value), dist_entropy, action_prob
    
class AvrgActorCritic():
    def __init__(self, max_n_hist, state_dim, action_dim, n_latent_var, type_dim = 0):
        self.nns = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_latent_var = n_latent_var
        self.type_dim = type_dim
        self.cur_n2add = 0
        self.max_n_hist = max_n_hist

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            # nn.ReLU(),
            nn.Tanh(),
            # nn.Linear(n_latent_var, n_latent_var),
            # nn.Tanh(),
            nn.Linear(n_latent_var, 1)
            )
    
    def add_to_hists(self, state_dict):
        if self.cur_n2add == len(self.nns):
            new_pol = ActorCritic(self.state_dim, self.action_dim, self.n_latent_var, self.type_dim).to(device)
            new_pol.load_state_dict(state_dict)
            self.nns.append(new_pol)
        else:
            self.nns[self.cur_n2add].load_state_dict(state_dict)
        self.cur_n2add = (self.cur_n2add + 1) % self.max_n_hist
    
    def act(self, state, memory):
        totl = len(self.nns)
        ret = None
        for i in range(totl):
            if i == 0:
                ret = self.nns[i].act(state, memory)
            else:
                ret = torch.add(ret, self.nns[i].act(state, memory))
        ret = ret / totl
        return ret
    
    def evaluate(self, state, action, typeob):
        totl = len(self.nns)
        ret = None
        for i in range(totl):
            if i == 0:
                ret0, _, ret1, ret2 = self.nns[i].evaluate(state, action, typeob)
            else:
                ret0_, _, ret1_, ret2_ = self.nns[i].evaluate(state, action, typeob)
                ret0 = torch.add(ret0, ret0_)
                ret1 = torch.add(ret1, ret1_)
                ret2 = torch.add(ret2, ret2_)
                # ret3 = torch.add(ret3, ret3_)

        ret0 = ret0 / totl
        ret1 = ret1 / totl
        ret2 = ret2 / totl
        # ret3 = ret3 / totl
        value_state = torch.cat([state], dim=1)
        state_value = self.value_layer(value_state)
        return ret0, torch.squeeze(state_value), ret1, ret2
    
    def state_dict(self):
        ret = []
        
        for i in range(len(self.nns)):
            ret.append(self.nns[i].state_dict())
        return [ret, self.value_layer.state_dict()]
    
    def load_state_dict(self, sdb):
        nns_sdb, value_sdb = sdb
        for i in range(len(self.nns)):
            self.nns[i].load_state_dict(nns_sdb[i])
        self.value_layer.load_state_dict(value_sdb)
        # return ret
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, minibatch, type_dim = 0, entcoeff = 0.01, value_lr=5e-2, entcoeff_decay = 0.99, max_n_hist = 10):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.minibatch = minibatch
        self.action_dim = action_dim
        self.max_n_hist = max_n_hist
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, type_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.action_layer.parameters(), lr=lr)
        # self.v_optimizer = torch.optim.Adam(self.policy.value_layer.parameters(), lr = value_lr)
        # self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)
        # self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, type_dim).to(device)
        # self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old = AvrgActorCritic(max_n_hist, state_dim, action_dim, n_latent_var, type_dim)
        self.policy_old.add_to_hists(self.policy.state_dict())
        self.v_old_opt = torch.optim.Adam(self.policy_old.value_layer.parameters(), lr=value_lr)
        # self.valuecoeff = valuecoeff
        self.value_lr = value_lr
        self.entcoeff = entcoeff
        self.entcoeff_decay = entcoeff_decay
        
        self.MseLoss = nn.MSELoss()
        # self.MseLoss = nn.SmoothL1Loss()
        self.policy_old_weight = 0
    
    def get_state_dict(self):
        return self.policy_old.state_dict()
    
    def set_state_dict(self, sdb):
        self.policy_old.load_state_dict(sdb)
    
    def v_update(self, memory, do_normalize=False):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            else:
                assert False
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # if rewards
        # rewards = torch.tensor(rewards).to(device)
        rewards = torch.stack(rewards).to(device)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device)
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        old_types = torch.stack(memory.type_obs).to(device).detach()

        # print attacker
        # if self.action_dim > 4:
        #     print('reward:')
        #     print(rewards[old_actions > 3][:20])
        #     print(old_states[old_actions > 3][:20])

        tot_value_loss = 0
        tot_loss = 0
        cnt_opt = 0

        for _ in range(self.K_epochs):
            for i_minibatch in range(math.ceil(rewards.shape[0] * 1. / self.minibatch)):
                _, state_values, _, _ = self.policy_old.evaluate(old_states[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch], old_actions[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch], old_types[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch])
                cur_value_loss = self.MseLoss(state_values, rewards[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch].detach())
                
                self.v_old_opt.zero_grad()
                cur_value_loss.mean().backward()
                self.v_old_opt.step()
                tot_value_loss += cur_value_loss.detach()
                cnt_opt += 1
        return tot_value_loss / cnt_opt
    
    def update(self, memory, do_normalize=False):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            else:
                assert False
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # if rewards
        # rewards = torch.tensor(rewards).to(device)
        rewards = torch.stack(rewards).to(device)

        # Normalizing the rewards:
        if do_normalize:
            # print(rewards)
            # print('do normalize!!!')
            # print(rewards.std())
            # print(torch.isnan(rewards).any())
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device)
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        old_types = torch.stack(memory.type_obs).to(device).detach()
        old_nextvs = torch.stack(memory.next_vs).to(device)

        # print('training batch:')
        # print(old_states)
        # print(old_actions)
        tot_value_loss = 0
        tot_loss = 0
        cnt_opt = 0

        # print('reward:')
        # print(rewards[:20])
        # print(old_types[old_actions > 3][:20])

        # print('reward shape')
        # print(rewards.shape[0])
        
        # Optimize policy for K epochs:
        # for _ in range(self.K_epochs):
        if True:
            # Evaluating old actions and values :
            # _, state_values, _, _ = self.policy.evaluate(old_states, old_actions, old_types)

            # state_values = torch.squeeze(state_values)
            
                
            # Finding Surrogate Loss:
            advantages = rewards# - state_values.detach()

            # if do_normalize:
                # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for i_minibatch in range(math.ceil(rewards.shape[0] * 1. / self.minibatch)):
                # print('i minibatch:')
                # print(i_minibatch, rewards.shape)
                # minibatch_idx = i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch
                cur_adv = advantages[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch]#.detach()

                _, _, logprobs, dist_entropy = self.policy.evaluate(old_states[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch], old_actions[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch], old_types[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch])
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch].detach())

                surr1 = ratios * cur_adv
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * cur_adv

                # cur_value_loss = self.MseLoss(state_values, rewards[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch].detach())

                # loss = -torch.min(surr1, surr2) - self.entcoeff*dist_entropy
                loss = logprobs * cur_adv.detach() + old_nextvs[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch]
                loss = -loss- self.entcoeff*dist_entropy


                self.entcoeff *= self.entcoeff_decay

                # print('cur value loss:')
                # print(loss)
                
                tot_loss += loss.mean().detach()
                cnt_opt += 1
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                break

        # Copy new weights into old policy:
        # self.policy_old.load_state_dict(self.policy.state_dict())
        # sdb = self.policy_old.state_dict()
        # sda = self.policy.state_dict()
        # for key in sda:
        #     sdb[key] = (sdb[key] * self.policy_old_weight + sda[key]) / (self.policy_old_weight + 1.)
        # self.policy_old_weight += 1
        # self.policy_old.load_state_dict(sdb)
        self.policy_old.add_to_hists(self.policy.state_dict())
        return tot_loss / cnt_opt

def calc_dis(a, b):
    # print('prepare to calcu dis:')
    # print(a)
    # print(b)
    if type(b) != torch.Tensor:
        try:
            b = torch.from_numpy(b).float().to(device)
        except:
            print(b)
            assert False
    ret = torch.norm(a - b)
    # print(ret)
    ret = max(ret, torch.tensor(1e-3))
    ret = ret ** (-2)
    # print(ret)
    if torch.isnan(ret).any():
        assert False
    # ret = max(ret, 1e-6)
    return ret

class NPAAgent:
    def __init__(self, n_step, n_belief, beliefs, *args, **kwargs):
        self.agents = []
        self.beliefs = []
        self.n_belief = n_belief
        self.n_target = beliefs[0][0].shape[0]
        self.n_step = n_step
        for i in range(n_step):
            tmp_agents = []
            for j in range(n_belief):
                tmp_agents.append(PPO(*args, **kwargs))
            self.agents.append(tmp_agents)
        
        # self.beliefs = beliefs

        # print(beliefs)
        self.beliefs = []
        for j in range(self.n_step):
            curbeliefs = []
            for i in range(self.n_belief):
                curbeliefs.append(torch.from_numpy(beliefs[j][i]).float().to(device))
            self.beliefs.append(curbeliefs)
        
        self.memory_ph = Memory()
    
    def get_state_dict(self):
        ret = []
        for i in range(self.n_step):
            tmp_ret = []
            for j in range(self.n_belief):
                tmp_ret.append(self.agents[i][j].get_state_dict())
            ret.append(tmp_ret)
        
        return ret
    
    def set_state_dict(self, s_dict):
        # print(s_dict)
        # print(self.n_step)
        # print(len(s_dict))
        # for i in range(self.n_step):
        for i in range(-self.n_step, 0):
            # print(i)
            cur_dict = s_dict[i]
            for j in range(self.n_belief):
                # print(i, j, type(cur_dict[j]))
                self.agents[i][j].set_state_dict(cur_dict[j])

    def load_model_with_specify(self, step, b, s_dict):
        self.agents[step][b].set_state_dict(s_dict[step][b])
    
    def v_update(self, sub_step, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # print('in v update:')
            # print(is_terminal, reward, discounted_reward)
            # print(type(reward))
            discounted_reward = reward + (self.agents[sub_step][memory.start_num].gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards).to(device)
        # print('reward before first step operation')
        # print(rewards)
        update_memory = memory.first_step(discounted_reward)

        ret = self.agents[sub_step][memory.start_num].v_update(memory.first_step(discounted_reward), False)
        update_memory.clear_memory()
        return ret

    def update(self, sub_step, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # print('error!!!')
            # print(type(reward))
            # print(reward)
            # print(type(discounted_reward))
            # print(sub_step, memory.start_num)

            discounted_reward = reward + (self.agents[sub_step][memory.start_num].gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards).to(device)
        # print('reward before first step operation')
        # print(rewards)
        update_memory = memory.first_step(discounted_reward)

        ret = self.agents[sub_step][memory.start_num].update(memory.first_step(discounted_reward), False)
        update_memory.clear_memory()
        return ret
    
    def get_w(self, step, ob):
        totd = 0
        # print(step, self.n_step)
        
        for i in range(self.n_belief):
            totd += calc_dis(self.beliefs[step][i], ob)
        
        ret = []
        for i in range(self.n_belief):
            ret.append(calc_dis(self.beliefs[step][i], ob) / totd) 

        return np.array(ret)
    
    def act(self, step, observation, memory, start_num = -1, in_training = False):
        # print(observation)

        if type(observation) != torch.Tensor:
            try:
                # print(observation)
                observation = torch.from_numpy(observation).float().to(device) 
            except:
                print(observation)
                assert False
        # print(observation)

        action_probs = None

        
        # print('in act:')

        if start_num == -1:
            # print('combine:')

            belief = observation[1:1 + self.n_target]

            # print('in act')
            # print(observation)
            # print(belief)

            w = self.get_w(step, belief)
            # print('w:')
            # print(w)

            first_enum = False
            for i in range(self.n_belief):
                if in_training:
                    action_prob = self.agents[step][i].policy.act(observation[1 + self.n_target:], memory)
                else:
                    action_prob = self.agents[step][i].policy_old.act(observation[1 + self.n_target:], memory)
                # print('step and action prob:')
                # print(step)
                # print(action_probs)
                # print(action_prob)

                if first_enum:
                    action_probs = torch.add(action_probs, action_prob * w[i])
                else:
                    action_probs = action_prob * w[i]
                    first_enum = True
                
                # print(action_probs)
                    
        else:
            # print('single')
            if in_training:
                action_probs = self.agents[step][start_num].policy.act(observation[1 + self.n_target:], memory)
            else:
                action_probs = self.agents[step][start_num].policy_old.act(observation[1 + self.n_target:], memory)
            memory.start_num = start_num
        
        # special mask for tag in tagging game
        # print('shape of action probs:')
        # print(action_probs)
        if action_probs.shape[0] > 4 and observation[-3] < 0.5:
            action_probs[-1] = 0
        
        action_probs = action_probs / torch.sum(action_probs)

        dist = Categorical(action_probs)
        action = dist.sample()
        
        # print(action_probs)
        # print(action)

        if memory != None:
            memory.states.append(observation[1 + self.n_target:])
            memory.actions.append(action.detach())
            memory.logprobs.append(dist.log_prob(action).detach())
        
        return action.item(), action_probs

    def evaluate(self, step, state, action, type_ob, start_num = -1, in_training = False):
        if start_num != -1:
            action_probs, state_values, _, _ = self.agents[step][start_num].policy.evaluate(state[:, self.n_target:], action, type_ob)
        else:
            # print('in eval:')
            # print(state)
            belief = state[0, 0: self.n_target]
            # print('in eval:')
            # print(state)
            # print(belief)
            w = self.get_w(step, belief)
            # print('weight:')
            # print(w)

            action_probs = None
            state_values = None
            first_enum = False
            for i in range(self.n_belief):
                if in_training:
                    action_prob, state_value, _, _ = self.agents[step][i].policy.evaluate(state[:, self.n_target:], action, type_ob)
                else:
                    # action_prob, _, _, _ = self.agents[step][i].policy_old.evaluate(state[:, self.n_target:], action, type_ob)
                    # _, state_value, _, _ = self.agents[step][i].policy.evaluate(state[:, self.n_target:], action, type_ob)
                    action_prob, state_value, _, _ = self.agents[step][i].policy_old.evaluate(state[:, self.n_target:], action, type_ob)
                # print('value:')
                # print(i, state_value, w[i], type_ob)
                # print('step and action prob:')
                # print(step)
                # print(action_probs)
                if first_enum:
                    action_probs = torch.add(action_probs, action_prob * w[i])
                    state_values = torch.add(state_values, state_value * w[i])
                else:
                    action_probs = action_prob * w[i]
                    state_values = state_value * w[i]
                    first_enum = True
        action_prob = action_probs[:, action]

        if type(action) != torch.Tensor:
            action = torch.tensor([action]).float().to(device)

        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)

        dist_entropy = dist.entropy()
        return  action_logprobs, torch.squeeze(state_values), dist_entropy, action_prob

class AtkNPAAgent:
    def __init__(self, n_type, *args, **kwargs):
        self.agents = [NPAAgent(*args, **kwargs) for _ in range(n_type)]
        # self.n_target = beliefs[0][0].shape[0]
        self.n_type = n_type
        self.memory_ph = Memory()
    
    def get_state_dict(self):
        ret = []
        for i in range(self.n_type):
            ret.append(self.agents[i].get_state_dict())
        return ret
    
    def set_state_dict(self, s_dict):
        for i in range(self.n_type):
            self.agents[i].set_state_dict(s_dict[i])
    
    def load_model_with_specify(self, step, belief, s_dict):
        for i in range(self.n_type):
            self.agents[i].load_model_with_specify(step, belief, s_dict[i])
    
    def update(self, sub_step, memory, type_n):
        return self.agents[type_n].update(sub_step, memory)
    
    def v_update(self, sub_step, memory, type_n):
        print('value of type {} now updating:'.format(type_n))
        return self.agents[type_n].v_update(sub_step, memory)

    def act(self, step, observation, memory, start_num = -1, type_n=None, in_training=False):
        if type_n != None:
            return self.agents[type_n].act(step, observation, memory, start_num, in_training)
        else:
            type_n = np.argmax(observation[-self.n_type:])
            return self.agents[type_n].act(step, observation, memory, start_num, in_training)

    def evaluate(self, step, state, action, type_n, type_ob, start_num=-1, in_training=False):
        return self.agents[type_n].evaluate(step, state, action, type_ob, start_num, in_training)
