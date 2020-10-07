import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.next_vs = []
        self.type_obs = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.next_vs[:]
        del self.type_obs[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, type_dim = 0):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                # nn.Linear(n_latent_var, n_latent_var),
                # nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                # nn.Linear(n_latent_var, n_latent_var),
                # nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().to(device) 

        action_probs = self.action_layer(state)

        if action_probs.shape[0] > 4 and state[-3] < 0.5:
            action_probs[-1] = 0
        
        action_probs = action_probs / torch.sum(action_probs)

        dist = Categorical(action_probs)
        action = dist.sample()
        
        if memory != None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
        
        return action.item(), action_probs
    
    def evaluate(self, state, action, typeob):
        if type(typeob) != torch.Tensor:
            typeob = torch.Tensor(typeob)

        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        
        if type(action) != torch.Tensor:
            action = torch.tensor([action]).float().to(device)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value_state = torch.cat([state], dim=1)
        
        state_value = self.value_layer(value_state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, type_dim=0):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # state dim need to +1 for round information
        
        self.policy = ActorCritic(state_dim + 1, action_dim, n_latent_var, type_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.action_layer.parameters(), lr=lr)
        # self.p_optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr, weight_decay=0.99)
        self.v_optimizer = torch.optim.Adam(self.policy.value_layer.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim + 1, action_dim, n_latent_var, type_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.memory_ph = Memory()
        self.policy_old_weight = 0
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        old_typeobs = torch.stack(memory.type_obs).to(device).detach()
        old_nextvs = torch.stack(memory.next_vs).to(device) # .detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_typeobs)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            # ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            # advantages = rewards + old_nextvs # - state_values.detach()
            # surr1 = ratios * advantages
            # surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # loss = -torch.min(surr1, surr2) # + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            loss = logprobs * rewards.detach() + old_nextvs # [i_minibatch * self.minibatch: (i_minibatch + 1) * self.minibatch]
            loss = -loss - 0.01*dist_entropy
            v_loss = self.MseLoss(state_values, rewards)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            self.v_optimizer.zero_grad()
            v_loss.mean().backward()
            self.v_optimizer.step()
        
        # Copy new weights into old policy:
        # self.policy_old.load_state_dict(self.policy.state_dict())
        sdb = self.policy_old.state_dict()
        sda = self.policy.state_dict()
        for key in sda:
            sdb[key] = (sdb[key] * self.policy_old_weight + sda[key]) / (self.policy_old_weight + 1.)
        self.policy_old_weight += 1
        self.policy_old.load_state_dict(sdb)
        # return tot_loss / cnt_opt
    
    def get_state_dict(self):
        return self.policy_old.state_dict()
    
    def set_state_dict(self, sdb):
        self.policy_old.load_state_dict(sdb)
    
    def evaluate(self, state, action, type_ob, in_training=False):
        if in_training:
            return self.policy.evaluate(state, action, type_ob)
        else:    
            return self.policy_old.evaluate(state, action, type_ob)
    
    def act(self, cur_round, state, memory, in_training=False):
        if in_training:
            return self.policy.act(state, memory)
        else:
            return self.policy_old.act(state, memory)

class AtkNPAAgent:
    def __init__(self, n_type, *args, **kwargs):
        self.agents = [PPO(*args, **kwargs) for _ in range(n_type)]
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
    
    def update(self, memory, type_n):
        return self.agents[type_n].update( memory)
    
    def act(self, cur_round, observation, memory, type_n=None, in_training=False):
        if type_n != None:
            return self.agents[type_n].act(cur_round, observation, memory, in_training)
        else:
            type_n = np.argmax(observation[-self.n_type:])
            return self.agents[type_n].act(cur_round, observation, memory, in_training)

    def evaluate(self, state, action, type_ob, type_n, in_training=False):
        return self.agents[type_n].evaluate(state, action, type_ob, in_training)