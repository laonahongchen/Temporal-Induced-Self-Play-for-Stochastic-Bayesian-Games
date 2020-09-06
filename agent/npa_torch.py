import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np
import gc
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.type_obs = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.type_obs[:]
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
                ret.is_terminals.append(self.is_terminals[i])
                ret.type_obs.append(self.type_obs[i])
            fetch = self.is_terminals[i]
        # ret.actions = self.actions[0:1]
        # ret.states = self.states[0:1]
        # ret.logprobs = self.logprobs[0:1]
        # ret.rewards = [reward]
        # ret.is_terminals = self.is_terminals[0:1]

        return ret


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, type_dim = 0):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                # nn.Linear(n_latent_var, n_latent_var),
                # nn.Tanh(),
                # nn.Linear(n_latent_var, n_latent_var),
                # nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim + type_dim, n_latent_var),
                nn.Tanh(),
                # nn.Linear(n_latent_var, n_latent_var),
                # nn.Tanh(),
                # nn.Linear(n_latent_var, n_latent_var),
                # nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def act(self, state, memory):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        
        return action_probs
    
    def evaluate(self, state, action, typeob):
        # print(type(state))
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        if type(typeob) != torch.Tensor:
            typeob = torch.from_numpy(typeob).float().to(device)
        # print('now in evaluate')
        # print(state.shape, typeob.shape)

        value_state = torch.cat([state, typeob], dim=1)
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
        state_value = self.value_layer(value_state)
        return action_probs, state_value, action_logprobs, dist_entropy # action_logprobs, torch.squeeze(state_value), dist_entropy, action_prob
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, minibatch, type_dim = 0, entcoeff = 0.01, valuecoeff=2, entcoeff_decay = 1.):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.minibatch = minibatch
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, type_dim).to(device)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, type_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.valuecoeff = valuecoeff
        self.entcoeff = entcoeff
        self.entcoeff_decay = entcoeff_decay
        
        self.MseLoss = nn.MSELoss()
        # self.MseLoss = nn.SmoothL1Loss()
        self.policy_old_weight = 0
    
    def update(self, memory, do_normalize=False):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(device)

        # Normalizing the rewards:
        if do_normalize:
            # print(rewards)
            # print('std:')
            # print(rewards.std())
            # print(torch.isnan(rewards).any())
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-3)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        old_types = torch.stack(memory.type_obs).to(device).detach()

        # print('training batch:')
        # print(old_states)
        # print(old_actions)
        tot_value_loss = 0
        tot_loss = 0
        cnt_opt = 0

        # print('reward shape')
        # print(rewards.shape[0])
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            _, state_values, _, _ = self.policy.evaluate(old_states, old_actions, old_types)

            state_values = torch.squeeze(state_values)
            
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()

            if do_normalize:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for i_minibatch in range(math.ceil(rewards.shape[0] * 1. / self.minibatch)):
                # print('i minibatch:')
                # print(i_minibatch, rewards.shape)
                # minibatch_idx = i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch
                cur_adv = advantages[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch].detach()

                _, state_values, logprobs, dist_entropy = self.policy.evaluate(old_states[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch], old_actions[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch], old_types[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch])
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch].detach())

                surr1 = ratios * cur_adv
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * cur_adv

                cur_value_loss = self.MseLoss(state_values[:, 0], rewards[i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch])

                loss = -torch.min(surr1, surr2) + self.valuecoeff*cur_value_loss - self.entcoeff*dist_entropy

                self.entcoeff *= self.entcoeff_decay

                # print('cur value loss:')
                # print(loss)
                tot_value_loss += cur_value_loss
                tot_loss += loss.mean()
                cnt_opt += 1
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
        # Copy new weights into old policy:
        # self.policy_old.load_state_dict(self.policy.state_dict())
        sdb = self.policy_old.state_dict()
        sda = self.policy.state_dict()
        for key in sda:
            sdb[key] = (sdb[key] * self.policy_old_weight + sda[key]) / (self.policy_old_weight + 1)
        self.policy_old_weight += 1
        self.policy_old.load_state_dict(sdb)
        return tot_value_loss / cnt_opt, tot_loss / cnt_opt

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

    def update(self, sub_step, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.agents[sub_step][memory.start_num].gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards).to(device)
        # print('reward before first step operation')
        # print(rewards)
        update_memory = memory.first_step(discounted_reward)

        ret = self.agents[sub_step][memory.start_num].update(memory.first_step(discounted_reward), True)
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
    
    def act(self, step, observation, memory, start_num = -1):
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
            action_probs = self.agents[step][start_num].policy_old.act(observation[1 + self.n_target:], memory)
            memory.start_num = start_num

        dist = Categorical(action_probs)
        action = dist.sample()
        
        # print(action_probs)
        # print(action)

        if memory != None:
            memory.states.append(observation[1 + self.n_target:])
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
        
        return action.item(), action_probs

    def evaluate(self, step, state, action, start_num, type_ob):
        if start_num != -1:
            action_probs, state_values, _, _ = self.agents[step][start_num].policy.evaluate(state[:, self.n_target:], action, type_ob)
        else:
            belief = state[0, 0: self.n_target]
            # print('in eval:')
            # print(state)
            # print(belief)
            w = self.get_w(step, belief)
            action_probs = None
            state_values = None
            first_enum = False
            for i in range(self.n_belief):
                action_prob, state_value, _, _ = self.agents[step][i].policy.evaluate(state[:, self.n_target:], action, type_ob)
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
    
    def update(self, sub_step, memory, type_n):
        return self.agents[type_n].update(sub_step, memory)

    def act(self, step, observation, memory, start_num = -1, type_n=None):
        if type_n != None:
            return self.agents[type_n].act(step, observation, memory, start_num)
        else:
            type_n = np.argmax(observation[-self.n_type:])
            return self.agents[type_n].act(step, observation, memory, start_num)

    def evaluate(self, step, state, action, start_num, type_n, type_ob):
        return self.agents[type_n].evaluate(step, state, action, start_num, type_ob)
