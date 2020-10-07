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

        self.lstm = nn.GRU(int(state_dim), int(n_latent_var))

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
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, rnn_history, memory):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state[1:]).float().to(device) 
        else:
            state = state[1:]
        # print('run gru:')
        # print(rnn_history.shape, state.shape)
        rec_state, new_rnn = self.lstm(state.unsqueeze(0).unsqueeze(0), rnn_history)
        # print('rec state:')
        # print(rec_state.shape, new_rnn.shape)
        action_probs = self.action_layer(rec_state.squeeze(0))

        # print(state[-3] < 0.5)
        # print(action_probs.shape)
        
        # print('before mask:')
        # print(action_probs)

        if action_probs.shape[1] > 4 and state[-3] < 0.5:
            action_probs[:, -1] = 0
        
        # print(action_probs)

        action_probs = action_probs / torch.sum(action_probs)
        
        dist = Categorical(action_probs)
        action = dist.sample()

        # print('action:', action)

        # print(action_probs, action)
        
        if memory != None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            memory.rec_states.append(rnn_history.squeeze(0).squeeze(0))
        
        return action.item(), new_rnn, action_probs
    
    def evaluate(self, state, rnn_history, action, type_ob):
        if type(state) != torch.Tensor:
            state = torch.Tensor(state)
        if type(rnn_history) != torch.Tensor:
            rnn_history = torch.Tensor(rnn_history)

        # print('in eval pre gru:')
        # print(state.unsqueeze(0).shape)
        rec_state, _ = self.lstm(state.unsqueeze(0), rnn_history)
        # print(rec_state)
        action_probs = self.action_layer(rec_state.squeeze(0).squeeze(0))
        dist = Categorical(action_probs)

        # print('action probs:')
        # print(action_probs)
        # print(action)
        
        if type(action) != torch.Tensor:
            action = torch.tensor([action]).float().to(device)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # print('in final eval:')
        # print(rec_state.shape)
        # print(type_ob.shape)

        rec_type_state = torch.cat([rec_state.squeeze(0)], dim=-1)
        
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
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        # self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, type_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # self.MseLoss = nn.MSELoss()
        self.MseLoss = nn.SmoothL1Loss()
        self.policy_old_weight = 0

        self.n_latent_var = n_latent_var

        self.rnn_history = torch.from_numpy(np.zeros((1, 1, n_latent_var))).float().to(device)
        self.memory_ph = Memory()
    
    def get_state_dict(self):
        return self.policy_old.state_dict()
    
    def set_state_dict(self, sdb):
        self.policy_old.load_state_dict(sdb)
    
    def update(self, memory, in_training=False):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device).detach()
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
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
            # print(old_type_obs.shape)
            # print(old_rnn_states.shape)

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_rnn_states.unsqueeze(0), old_actions, old_type_obs)
            
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
    def act(self, state, memory=None, history=None, in_training=False):
        if type(history) != torch.Tensor:
            history = self.rnn_history
        if memory == None:
            memory = self.memory_ph
        if in_training:
            ac, self.rnn_history, acs = self.policy.act(state, history, memory)
        else:
            ac, self.rnn_history, acs = self.policy_old.act(state, history, memory)
        self.memory_ph.clear_memory()
        return ac, self.rnn_history, acs
    
    def rnn_reset(self):
        self.rnn_history = torch.from_numpy(np.zeros((1, 1, self.n_latent_var))).float().to(device)
        # self.memory_ph.clear_memory()

    def evaluate(self, state, rnn_history, action, type_ob, in_training=False):
        # print('in eval func:')
        # print(rnn_history.shape)
        # print(type_ob.shape)
        
        if type(type_ob) != torch.Tensor:
            type_ob = torch.Tensor(type_ob)

        if in_training:
            return self.policy.evaluate(state, rnn_history, action, type_ob.unsqueeze(0))
        else:
            return self.policy_old.evaluate(state, rnn_history, action, type_ob.unsqueeze(0))


class AtkNPAAgent:
    def __init__(self, n_type, *args, **kwargs):
        self.agents = [PPO(*args, **kwargs) for _ in range(n_type)]
        # self.n_target = beliefs[0][0].shape[0]
        self.n_type = n_type
        self.memory_ph = Memory()
        
        self.rnn_history = torch.zeros_like(self.agents[0].rnn_history).float().to(device)
    
    def rnn_reset(self):
        # self.rnn_history = torch.from_numpy(np.zeros((1, 1, self.n_latent_var))).float().to(device)
        for i in range(self.n_type):
            self.agents[i].rnn_reset()

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
    
    def update(self, memory, type_n):
        return self.agents[type_n].update(memory)

    def act(self, state, memory=None, history=None,type_n=None, in_training=False):
        if type(history) != torch.Tensor:
            history = self.rnn_history
        if memory == None:
            memory = self.memory_ph
        if type_n != None:
            return self.agents[type_n].act(state, memory, history, in_training)
        else:
            type_n = np.argmax(state[-self.n_type:])
            return self.agents[type_n].act(state, memory, history, in_training)

    def evaluate(self, state, rnn_history, action, type_ob, type_n, in_training=False):
        return self.agents[type_n].evaluate(state, rnn_history, action, type_ob, in_training)