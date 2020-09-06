import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np
import gc
import math
from agent.npa_torch import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                # nn.Linear(n_latent_var, n_latent_var),
                # nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                # nn.Linear(n_latent_var, n_latent_var),
                # nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def act(self, state):
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

        if type(action) != torch.Tensor:
            action = torch.tensor([action]).to(device)

        state_value = self.value_layer(value_state) * action
        return action_probs, state_value # action_logprobs, torch.squeeze(state_value), dist_entropy, action_prob

def calc_dis(a, b):
    if type(b) != torch.Tensor:
        try:
            b = torch.from_numpy(b).float().to(device)
        except:
            print(b)
            assert False
    ret = torch.norm(a - b)
    ret = max(ret, torch.tensor(1e-3))
    ret = ret ** (-2)
    if torch.isnan(ret).any():
        assert False
    return ret

class SampleACAgent:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, k_epochs, print_every=100):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.action_dim = action_dim
        # self.eps_clip = eps_clip
        self.K_epochs = k_epochs
        # self.minibatch = minibatch
        
        # self.policy = ActorCritic(state_dim, action_dim, n_latent_var, type_dim).to(device)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        # self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        # self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_optimizer = torch.optim.Adam(self.policy_old.value_layer.parameters(), lr=lr, betas = betas)
        self.policy_optimizer = torch.optim.Adam(self.policy_old.action_layer.parameters(), lr=lr, betas=betas)
        
        # self.MseLoss = nn.MSELoss()
        # self.MseLoss = nn.SmoothL1Loss()
        self.value_loss = nn.SmoothL1Loss()
        self.pol_loss = nn.MSELoss()

        self.k_epochs = k_epochs
        self.print_every = print_every
    
    def value_supervise(self, value_state, value_pred):
        if type(value_state) != torch.Tensor:
            # value_state = torch.from_numpy(value_state).float().to(device)
            value_state = torch.Tensor(value_state).float().to(device)
        if type(value_pred) != torch.Tensor:
            value_pred = torch.Tensor(value_pred).float().to(device)
            # value_pred = torch.from_numpy(value_pred).float().to(device)
        for t in range(self.k_epochs):
            y_pred = self.policy_old.value_layer(value_state)
            loss = self.value_loss(y_pred, value_pred)
            if t % self.print_every == self.print_every - 1 or self.print_every == 1:
                print('episode: {}, loss: {}'.format(t, loss))
            
            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()
    
    def policy_supervise(self, obs, policys):
        if type(obs) != torch.Tensor:
            obs = torch.Tensor(obs).float().to(device)
        # print(policys)
        # print(type(policys))
        if type(policys) != torch.Tensor:
            # policys = torch.Tensor(policys).float().to(device)
            policys = torch.stack(policys).float().to(device)
        for t in range(self.k_epochs):
            y_pred = self.policy_old.action_layer(obs)
            loss = self.pol_loss(y_pred, policys)
            if t % self.print_every == self.print_every - 1 or self.print_every == 1:
                print('episode: {}, loss: {}'.format(t, loss))
            
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

    def policy_update(self, state, opponent_action_prob, cur_values):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        
        # print('state:')
        # print(state, state.shape)
        # print(cur_values.shape)

        self_action = self.policy_old.action_layer(state)
        opponent_dim = opponent_action_prob.shape[0]
        # tot_value = [0 for _ in range(state.shape[0])]
        tot_value = 0
        for i in range(self.action_dim):
            for j in range(opponent_dim):
                # print(type(cur_values), self_action)
                # print(cur_values)
                # print(cur_values[i, j], self_action[i], opponent_action_prob[j])
                add_v = cur_values[i, j] * self_action[i] * opponent_action_prob[j]
                # print(add_v)
                # print(type(add_v))
                # tot_value += cur_values[i, j] * self_action[i] * opponent_action_prob[j]
                tot_value += add_v
        loss = -tot_value
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()


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
                tmp_agents.append(SampleACAgent(*args, **kwargs))
            self.agents.append(tmp_agents)
        
        # self.beliefs = beliefs

        # print(beliefs)
        self.beliefs = []
        for j in range(self.n_step):
            curbeliefs = []
            for i in range(self.n_belief):
                curbeliefs.append(torch.from_numpy(beliefs[j][i]).float().to(device))
            self.beliefs.append(curbeliefs)
    
    def get_w(self, step, ob):
        totd = 0
        # print(step, self.n_step)
        
        for i in range(self.n_belief):
            totd += calc_dis(self.beliefs[step][i], ob)
        
        ret = []
        for i in range(self.n_belief):
            ret.append(calc_dis(self.beliefs[step][i], ob) / totd) 

        return np.array(ret)
        
        # self.memory_ph = Memory()
    
    def act(self, step, observation, start_num = 1):
        if type(observation) != torch.Tensor:
            observation = torch.from_numpy(observation).float().to(device) 

        action_probs = None

        if start_num == -1:
            belief = observation[1:1 + self.n_target]

            w = self.get_w(step, belief)

            first_enum = False
            for i in range(self.n_belief):
                action_prob = self.agents[step][i].policy_old.act(observation[1 + self.n_target:])

                if first_enum:
                    action_probs = torch.add(action_probs, action_prob * w[i])
                else:
                    action_probs = action_prob
                    first_enum = True
                    
        else:
            action_probs = self.agents[step][start_num].policy_old.act(observation[1 + self.n_target:])

        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), action_probs

    def evaluate(self, step, observation, action, start_num = -1):
        if type(observation) != torch.Tensor:
            observation = torch.from_numpy(observation).float().to(device) 
        if start_num == -1:
            belief = observation[:,:self.n_target]
            w = self.get_w(step, belief)

            first_enum = False
            
            for i in range(self.n_belief):
                action_prob = self.agents[step][i].policy_old.act(observation[:,self.n_target:])

                if first_enum:
                    action_probs = torch.add(action_probs, action_prob * w[i])
                    v_preds = torch.add(v_preds, self.agents[step][i].policy_old.value_layer(observation[:,self.n_target:]) * w[i])
                else:
                    action_probs = action_prob * w[i]
                    v_preds = self.agents[step][i].policy_old.value_layer(observation[self.n_target:]) * w[i]
                    first_enum = True
            
            return action_probs[action], v_preds
        else:
            # print(self.n_target, observation.shape)
            action_probs = self.agents[step][start_num].policy_old.act(observation[:,self.n_target:])
            v_pred = self.agents[step][start_num].policy_old.value_layer(observation[:,self.n_target:])
            # print(action_probs)
            return action_probs[:, action], v_pred
    
    def update(self, step, state, opponent_strategy, start_num, vs):
        # print(opponent_strategy.shape)
        # print(state.shape)
        self.agents[step][start_num].policy_update(state[self.n_target + 1:], opponent_strategy, vs)
    
    def policy_supervise(self, step, start_num, obs, pols):
        
        self.agents[step][start_num].policy_supervise(obs, pols)
    
    def value_supervise(self, step, start_num, obs, v_preds):
        self.agents[step][start_num].value_supervise(obs, v_preds)


class AtkNPAAgent:
    def __init__(self, n_type, *args, **kwargs):
        self.agents = [NPAAgent(*args, **kwargs) for _ in range(n_type)]
        self.n_type = n_type
        self.memory_ph = Memory()
    
    def update(self, step, state, opponent_strategy, start_num, vs):
        # return self.agents[type_n].update(sub_step, memory)
        for type_i in range(self.n_type):
            self.agents[type_i].update(step, state, opponent_strategy, start_num, vs[type_i])

    def act(self, step, observation, start_num = -1, type_n=None):
        if type_n != None:
            return self.agents[type_n].act(step, observation, start_num)
        else:
            type_n = np.argmax(observation[-self.n_type:])
            return self.agents[type_n].act(step, observation, start_num)

    def evaluate(self, step, state, action, start_num, type_n):
        return self.agents[type_n].evaluate(step, state, action, start_num)

    def policy_supervise(self, step, start_num, obs, pols):
        for type_i in range(self.n_type):
            self.agents[type_i].policy_supervise(step, start_num, obs[type_i], pols[type_i])

    def value_supervise(self, step, start_num, obs, v_preds):
        for type_i in range(self.n_type):
            self.agents[type_i].value_supervise(step, start_num, obs[type_i], v_preds[type_i])