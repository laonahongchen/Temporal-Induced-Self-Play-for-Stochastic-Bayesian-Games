import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
    
    def first_step(self, reward):
        ret = Memory()
        ret.actions = self.actions[0:1]
        ret.states = self.states[0:1]
        ret.logprobs = self.logprobs[0:1]
        ret.rewards = [reward]
        ret.is_terminals = self.is_terminals[0:1]

        return ret



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # print('state dim:')
        # print(action_dim)

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().to(device) 
        # print('state & actions:')
        # print(state)
        action_probs = self.action_layer(state)
        # print(action_probs)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        if memory != None:
            # print('act state:')
            # print(state)
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        # print(type(state))
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        # else:
            # print(state.device)
        # print('state 1:')
        # print(state)
        action_probs = self.action_layer(state)

        # print('state 2:')
        # print(state)

        # print(action_probs)

        action_prob = action_probs[:, action]

        # print('state 3:')
        # print(state)

        if type(action) != torch.Tensor:
            action = torch.tensor([action]).float().to(device)

        # print('state:')
        # print(state)

        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)

        # print(action_logprobs)

        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)
        # print('state value:')
        # print(state_value.cpu().detach())
        return action_logprobs, torch.squeeze(state_value), dist_entropy, action_prob
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        # print('state dim:')
        # print(state_dim)
    
    def update(self, memory, do_normalize=False):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # print('rewards:')
        # print(rewards)
        # print(memory.states)
        # print(memory.actions)
        # print(memory.logprobs)

        rewards = torch.tensor(rewards).to(device)

        # Normalizing the rewards:
        if do_normalize:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :

            # print('old state:')
            # print(old_states)

            logprobs, state_values, dist_entropy, _ = self.policy.evaluate(old_states, old_actions)

            # print(state_values.cpu().detach())
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # print("prob:")
            # print(surr1.device)
            # print(surr2.device)
            # print(state_values)
            # print(rewards)
            # print(dist_entropy.device)
            # print(surr1, surr2, state_values)
            # print(rewards, dist_entropy)
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # print('prepare to optimize')
            # print(loss)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            # torch.nn.utils.clip_grad_norm(self.policy.parameters(), 0.1)
            self.optimizer.step()

            # print('optimize finish')
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

class BackInductionAgent:
    def __init__(self, n_step, *args, **kwargs):
        self.agents = []
        for i in range(n_step):
            self.agents.append(PPO(*args, **kwargs))

    def update(self, sub_step, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.agents[sub_step].gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards).to(device)

        self.agents[sub_step].update(memory.first_step(discounted_reward))
    
    def act(self, step, observation, memory):
        # print('obs:')
        # print(observation)
        return self.agents[step].policy_old.act(observation[1:], memory)
    
    def evaluate(self, step, state, action):
        return self.agents[step].policy.evaluate(state, action)

