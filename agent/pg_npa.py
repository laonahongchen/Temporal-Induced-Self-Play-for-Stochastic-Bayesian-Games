
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, type_dim = 0):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
        # self.logprobs = []
        # self.state_values = []
        # self.rewards = []

    def forward(self, state):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float()
        # state = F.relu(self.affine(state))
        
        state_value = self.value_layer(state)
        
        action_probs = self.action_layer(state)
        # action_distribution = Categorical(action_probs)
        # action = action_distribution.sample()
        
        # self.logprobs.append(action_distribution.log_prob(action))
        # self.state_values.append(state_value)
        
        return action_probs
    
    def calculateLoss(self, rewards, logprobs, state_values):        
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        # rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        # for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
        for logprob, value, reward in zip(logprobs, state_values, rewards):
            # print('in cal loss:')
            # print(logprob, value, reward)
            # advantage = reward  - value.item()
            # action_loss = -logprob * advantage
            action_loss = -torch.exp(logprob) * reward
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   
        return loss
    
    def evaluate(self, state, action):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        # if type(typeob) != torch.Tensor:
            # typeob = torch.from_numpy(typeob).float().to(device)
        
        # value_state = torch.cat([state, typeob], dim=1)
        action_probs = self.action_layer(state)

        action_prob = action_probs[:, action]

        if type(action) != torch.Tensor:
            action = torch.tensor([action]).to(device)

        dist = Categorical(action_probs)
        # action_logprobs = dist.log_prob(action)
        # dist_entropy = dist.entropy()

        state_value = self.value_layer(state)
        return action_probs, torch.squeeze(state_value) #, action_logprobs, dist_entropy
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]


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

class PGAgent:
    def __init__(self, lr, betas, *args, **kwargs):        
        self.policy = ActorCritic(*args, **kwargs)
        self.policy_old = ActorCritic(*args, **kwargs)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old_weight = 0
    
    def update(self, rewards, logprobs, state_values):

        loss = self.policy.calculateLoss(rewards, logprobs, state_values)
        # print('loss:')
        # print(loss)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        sdb = self.policy_old.state_dict()
        sda = self.policy.state_dict()
        for key in sda:
            sdb[key] = (sdb[key] * self.policy_old_weight + sda[key]) / (self.policy_old_weight + 1.)
        self.policy_old_weight += 1
        self.policy_old.load_state_dict(sdb)

    def act(self, state, in_training = False):
        if in_training:
            return self.policy(state), self.policy.value_layer(state)
        else:
            return self.policy_old(state), self.policy.value_layer(state)
    
    def evaluate(self, state, action, in_training = False):
        if in_training:
            return self.policy.evaluate(state, action)
        else:
            return self.policy_old.evaluate(state, action)

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
                tmp_agents.append(PGAgent(*args, **kwargs))
            self.agents.append(tmp_agents)
        
        self.beliefs = []
        for j in range(self.n_step):
            curbeliefs = []
            for i in range(self.n_belief):
                curbeliefs.append(torch.from_numpy(beliefs[j][i]).float().to(device))
            self.beliefs.append(curbeliefs)
        
        # self.memory_ph = Memory()
        self.start_num = -1
        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def update(self, sub_step, gamma = 0.99):
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        # update_memory = memory.first_step(discounted_reward)

        ret = self.agents[sub_step][self.start_num].update(rewards[:1], self.logprobs[-1:], self.state_values[-1:])
        # update_memory.clear_memory()
        return ret
    
    def get_w(self, step, ob):
        totd = 0
        
        for i in range(self.n_belief):
            totd += calc_dis(self.beliefs[step][i], ob)
        
        ret = []
        for i in range(self.n_belief):
            ret.append(calc_dis(self.beliefs[step][i], ob) / totd) 

        return np.array(ret)
    
    def act(self, step, observation, start_num = -1, in_training = False):
        if type(observation) != torch.Tensor:
            try:
                observation = torch.from_numpy(observation).float().to(device) 
            except:
                print(observation)
                assert False

        action_probs = None

        if start_num == -1:
            belief = observation[1:1 + self.n_target]
            w = self.get_w(step, belief)

            first_enum = False
            for i in range(self.n_belief):
                # if in_training:
                #     action_prob, vs = self.agents[step][i].policy.act(observation[1 + self.n_target:])
                # else:
                #     action_prob, vs = self.agents[step][i].policy_old.act(observation[1 + self.n_target:])
                action_prob, vs = self.agents[step][i].act(observation[1 + self.n_target:], in_training)

                if first_enum:
                    action_probs = torch.add(action_probs, action_prob * w[i])
                else:
                    action_probs = action_prob * w[i]
                    first_enum = True
                    
        else:
            # if in_training:
            #     action_probs, vs = self.agents[step][start_num].policy.act(observation[1 + self.n_target:], memory)
            # else:
            #     action_probs, vs = self.agents[step][start_num].policy_old.act(observation[1 + self.n_target:], memory)
            # memory.start_num = start_num
            self.start_num = start_num
            action_probs, vs = self.agents[step][start_num].act(observation[1 + self.n_target:], in_training)

        # print('probs:')
        # print(action_probs)
        dist = Categorical(action_probs)
        action = dist.sample()

        self.logprobs.append(dist.log_prob(action))
        self.state_values.append(vs)
        
        return action.item(), action_probs

    def evaluate(self, step, state, action, start_num = -1, in_training = False):
        if start_num != -1:
            action_probs, state_values = self.agents[step][start_num].policy.evaluate(state[:, self.n_target:], action)
        else:
            belief = state[0, 0: self.n_target]
            w = self.get_w(step, belief)
            action_probs = None
            state_values = None
            first_enum = False
            for i in range(self.n_belief):
                action_prob, state_value = self.agents[step][i].policy.evaluate(state[:, self.n_target:], action)
                if first_enum:
                    action_probs = torch.add(action_probs, action_prob * w[i])
                    state_values = torch.add(state_values, state_value * w[i])
                else:
                    action_probs = action_prob * w[i]
                    state_values = state_value * w[i]
                    first_enum = True
        action_prob = action_probs[:, action]

        # if type(action) != torch.Tensor:
            # action = torch.tensor([action]).float().to(device)

        # dist = Categorical(action_probs)
        
        # action_logprobs = dist.log_prob(action)

        # dist_entropy = dist.entropy()
        # return  action_logprobs, torch.squeeze(state_values), dist_entropy, action_prob
        return torch.squeeze(state_values), action_prob

class AtkNPAAgent:
    def __init__(self, n_type, *args, **kwargs):
        self.agents = [NPAAgent(*args, **kwargs) for _ in range(n_type)]
        self.n_type = n_type
        # self.memory_ph = Memory()
    
    def update(self, sub_step, type_n, gamma=0.99):
        return self.agents[type_n].update(sub_step)

    def act(self, step, observation, start_num = -1, type_n=None, in_training=False):
        if type_n != None:
            return self.agents[type_n].act(step, observation, start_num, in_training)
        else:
            type_n = np.argmax(observation[-self.n_type:])
            return self.agents[type_n].act(step, observation, start_num, in_training)

    def evaluate(self, step, state, action, type_n, start_num=-1, in_training=False):
        return self.agents[type_n].evaluate(step, state, action, start_num, in_training)
    
