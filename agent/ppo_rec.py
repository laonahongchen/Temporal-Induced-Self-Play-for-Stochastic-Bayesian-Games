import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.rec_states = []
        self.type_obs = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.rec_states[:]
        del self.type_obs[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, type_dim):
        super(ActorCritic, self).__init__()

        # print(type(state_dim), n_latent_var)

        self.gru = nn.GRU(int(state_dim), int(n_latent_var))

        # actor
        self.action_layer = nn.Sequential(
                # nn.Linear(n_latent_var, n_latent_var),
                # nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                # nn.Linear(n_latent_var, n_latent_var),
                # nn.Tanh(),
                nn.Linear(n_latent_var + type_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, rnn_history, memory):
        state = torch.from_numpy(state[1:]).float().to(device) 
        # print('run gru:')
        # print(rnn_history, state)
        rec_state, new_rnn = self.gru(state.unsqueeze(0).unsqueeze(0), rnn_history)
        # print('rec state:')
        # print(rec_state)
        action_probs = self.action_layer(rec_state.squeeze(0))
        # print(action_probs)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        if memory != None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            memory.rec_states.append(rnn_history.squeeze(0).squeeze(0))
        
        return action.item(), new_rnn, action_probs
    
    def evaluate(self, state, rnn_history, action, type_ob):
        # print('in eval:')
        # print(rnn_history.shape)
        rec_state, _ = self.gru(state.unsqueeze(0), rnn_history.unsqueeze(0))
        # print(rec_state)
        action_probs = self.action_layer(rec_state.squeeze(0))
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # print('in eval:')
        # print(rec_state.shape)
        # print(type_ob.shape)

        rec_type_state = torch.cat([rec_state, type_ob.unsqueeze(0)], dim=2)
        
        state_value = self.value_layer(rec_type_state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, type_dim):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, type_dim).to(device)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, type_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # self.MseLoss = nn.MSELoss()
        self.MseLoss = nn.SmoothL1Loss()
        self.policy_old_weight = 0

        self.n_latent_var = n_latent_var

        self.rnn_history = torch.from_numpy(np.zeros((1, 1, n_latent_var))).float().to(device)
        self.memory_ph = Memory()
    
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
        old_rnn_states = torch.stack(memory.rec_states).to(device).detach()
        old_type_obs = torch.stack(memory.type_obs).to(device).detach()

        # print('in update:')
        # print(old_states)
        # print(old_rnn_states)
        tot_value_loss = 0
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_rnn_states, old_actions, old_type_obs)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            value_loss = self.MseLoss(state_values, rewards)
            tot_value_loss += value_loss
            
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
        return tot_value_loss / self.K_epochs

    # act function for test
    def act(self, state, history=None):
        if history == None:
            history = self.rnn_history
        ac, self.rnn_history, acs = self.policy.act(state, history, self.memory_ph)
        self.memory_ph.clear_memory()
        return ac, self.rnn_history, acs
    
    def rnn_reset(self):
        self.rnn_history = torch.from_numpy(np.zeros((1, 1, self.n_latent_var))).float().to(device)
        # self.memory_ph.clear_memory()
