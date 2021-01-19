from env.base_env import BaseEnv
from gym import spaces
import numpy as np
import torch

def calc_dis(a, b):
        x1, y1 = a
        x2, y2 = b
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.2

def one_hot(n, i):
    x = np.zeros(n)
    if i < n:
        x[i] = 1.
    return x

def n_one_hot(n, a):
    ret = np.array([])
    for i in range(a.shape[0]):
        # print(i)
        ret = np.concatenate((ret, one_hot(n, int(a[i]))))
    return ret

class TaggingEnv(BaseEnv):
    def __init__(self, prior, n_steps, is_atk=False, seed=None):
        # self.zero_sum = True
        # self.payoff = [[1., -1.], [-1., 1.]]

        # self.zero_sum = False
        # self.payoff = [[[-1., -1.], [-3., 0.]], [[0., -3.], [-2., -2.]]]

        self.v = [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0], [0, 0]]
        self.target = [[7, 0], [7, 7]]
        self.n_types = self.n_targets = 2
        self.n_rounds = self.n_steps = n_steps
        self.n_agents = 2
        self.prior = prior
        self.is_atk = is_atk
        self.random_prior = False
        self.seed = seed

        self.grids = np.zeros((8, 8))
        self.zero_sum = True

        if seed is not None:
            np.random.seed(int(seed))
        # self.debug
        # self.obs_size = obs_size
        
        # For one cell, the possible result is nothing, fruits, either agents, both agents
        # The last observation for the attacker is its goal
        # observation_spaces = [spaces.Tuple(spaces.Box(low=0., high=8., shape=[4]), spaces.Box(low=0., high=1., shape=[4])), spaces.Tuple(spaces.Box(low=0., high=8., shape=[2]), spaces.Box(low=0., high=1., shape=[4]))]
        observation_spaces = [spaces.Box(low=0., high=8., shape=[9]), spaces.Box(low=0., high=8., shape=[9])]

        action_spaces = [spaces.Discrete(4), spaces.Discrete(5)]

        super().__init__(2, observation_spaces, action_spaces)                                                              

        self.reset()
    
    def get_atk_env(self):
        return TaggingEnv(self.prior, self.n_steps, True)
    
    def set_prior(self, prior):
        self.prior = prior

    def update_policy(self, i, policy):
        if i == 0:
            self.atk_policy = policy
        else:
            self.dfd_policy = policy

    def _get_obs(self):
        x1, y1 = self.att_p
        x2, y2 = self.def_p

        atk_obs = np.array([self.att_p, self.def_p, [x1 - x2, y1 - y2], [1 if calc_dis(self.def_p, self.att_p) <= 6.25 ** 0.2 and self.att_p[0] < self.grids.shape[0] / 2 else 0, 0]]).reshape(-1)[:-1]
        def_obs = np.copy(atk_obs)
        # print(atk_obs)
        # print()
        # atk_obs = self.grids.reshape(-1)
        # atk_obs = n_one_hot(4, atk_obs)
        # def_obs = self.grids.reshape(-1)
        # def_obs = n_one_hot(4, def_obs)
        
        # atk_obs = np.concatenate((np.array([self.round_cnt]), self.belief, atk_obs, one_hot(self.n_targets, self.goal)))
        atk_obs = torch.cat((torch.Tensor([self.round_cnt]), torch.Tensor(self.belief), torch.Tensor(atk_obs), torch.Tensor(one_hot(self.n_targets, self.goal))))
        def_obs = torch.cat((torch.Tensor([self.round_cnt]), torch.Tensor(self.belief), torch.Tensor(def_obs), torch.zeros(self.n_targets)))
        # if np.random.rand() < 0.8:
        #     # def_obs = np.concatenate((np.array([self.round_cnt]), self.belief, def_obs, one_hot(self.n_targets, self.goal)))
        #     def_obs = torch.cat((torch.Tensor([self.round_cnt]), torch.Tensor(self.belief), torch.Tensor(def_obs), torch.Tensor(one_hot(self.n_targets, self.goal))))
        # else:
        #     def_obs = torch.cat((torch.Tensor([self.round_cnt]), torch.Tensor(self.belief), torch.Tensor(def_obs), torch.Tensor(one_hot(self.n_targets, 1 - self.goal))))
            # def_obs = np.concatenate((np.array([self.round_cnt]), self.belief, one_hot(self.n_targets, 1 - self.goal), def_obs))
        
        # return [np.array(atk_obs), np.array(def_obs)]
        return atk_obs, def_obs

    def get_type_space(self, i):
        return spaces.Discrete(self.n_targets)

    def generate_prior(self):
        x = [0.] + sorted(np.random.rand(self.n_targets - 1).tolist()) + [1.]
        prior = np.zeros(self.n_targets)
        for i in range(self.n_targets):
            prior[i] = x[i + 1] - x[i]
        return prior

    def reset(self, debug=False, verbose=False):
        self.round_cnt = 0
        self.prob_cnt = 0
        self.att_p = [0, 4]
        self.def_p = [np.random.randint(4), np.random.randint(8)]
        if self.random_prior:
            self.prior = self.generate_prior()
        if self.is_atk:
            self.atk_type = self.goal = self.type = np.random.choice(self.n_targets)
        else:
            self.atk_type = self.goal = self.type = np.random.choice(self.n_targets, p=self.prior)
        
        # self.belief = np.copy(self.prior)
        self.belief = torch.Tensor(self.prior)
        self.last_obs_n = self._get_obs()
        self.debug = debug
        # if verbose or debug:
            # print('atk pos:')
            # print(self.att_p)
            # print('def pos:')
            # print(self.def_p)
        return self.last_obs_n, None, None
    
    def sub_reset(self, round, belief, debug=False, verbose=False):
        # print('subreset')

        self.round_cnt = round
        self.prob_cnt = 0
        self.att_p = [np.random.randint(8), np.random.randint(8)]
        self.def_p = [np.random.randint(4), np.random.randint(8)]
        self.atk_type = self.goal = self.type = np.random.choice(self.n_targets, p=belief)

        # print(self.att_p, self.def_p)
        
        # self.belief = np.copy(belief)
        self.belief = torch.Tensor(belief)
        self.last_obs_n = self._get_obs()
        # print(self.last_obs_n[0].shape)
        self.debug = debug
        # if verbose or debug:
            # print('atk pos:')
            # print(self.att_p)
            # print('def pos:')
            # print(self.def_p)
        return self.last_obs_n, None, None

    
    # use together with reset_to_state() function
    def get_cur_state(self):
        # print('get cur state:')
        # print(self.att_p, self.def_p)
        return (np.copy(self.att_p), np.copy(self.def_p)), self.belief
    
    def reset_to_state(self, round, pos, belief):
        # print('reset to state')
        # print(pos)
        self.round_cnt = round
        self.att_p, self.def_p = np.copy(pos)
        self.atk_type = self.goal = self.type = np.random.choice(self.n_targets, p=belief)
        # self.type_ob = np.zeros(shape=self.n_types, dtype=np.float32)
        # self.type_ob[self.atk_type] = 1.
        self.belief = belief
        self.last_obs_n = self._get_obs()
        return self.last_obs_n, None, None # , self.probs, self.ac_history
    
    def reset_to_state_with_type(self, round, pos, belief, type_s):
        # print('reset to state with type')
        # print(pos)
        self.round_cnt = round
        self.att_p, self.def_p = np.copy(pos)
        self.atk_type = self.goal = self.type = type_s # np.random.choice(self.n_targets, p=belief)
        # self.type_ob = np.zeros(shape=self.n_types, dtype=np.float32)
        # self.type_ob[self.atk_type] = 1.
        self.belief = belief
        self.last_obs_n = self._get_obs()
        return self.last_obs_n, None, None # self.probs, self.ac_history

    def update_belief(self, belief, probs):
        if type(probs) == torch.Tensor:
            if type(belief) != torch.Tensor:
                belief = torch.Tensor(belief)
            probs = probs.reshape(-1)
            tmp = belief * probs
            if torch.sum(tmp) < 1e-2:
                return torch.ones(self.n_targets) / self.n_targets
            return tmp / torch.sum(tmp)
        else:
            probs = probs.reshape(-1)
            tmp = belief * probs
            if np.sum(tmp) < 1e-2:
                return np.ones(self.n_targets) / self.n_targets
            return tmp / np.sum(tmp)
    
    def _get_atk_ob(self, target, belief, last_obs_atk) :
        ret = np.copy(last_obs_atk)
        # print('atk ob shape')
        # print(ret.shape)
        obs_round = ret[0]
        ret = ret[1 + self.env.n_targets: -self.env.n_targets]
        ret = np.concatenate((obs_round, belief, ret, one_hot(self.n_targets, target)))
        # ret[1] = belief[0]
        # ret[2] = belief[1]
        # print(ret.shape)
        return ret
    
    def update_policy(self, i, policy):
        if i == 0:
            self.atk_policy = policy
        else:
            self.dfd_policy = policy

    def step(self, actions, probs=None, verbose=False):
        if actions[1] == 4 and not (calc_dis(self.att_p, self.def_p) <= 6.25 ** 0.2 and self.att_p[0] < self.grids.shape[0] / 2):
            # actions[1] = 0
            print(self._get_obs())
            assert False

        if verbose or self.debug:
            print('atk pos:')
            print(self.att_p)
            print('def pos:')
            print(self.def_p)
            print('actions')
            print(actions)

        # although the belief part is definitely should be executed in agent part and is agent-wise 
        # for simplicity in implentation, this is implement here
        # and this is not a problem bigger that the sub_reset function

        # probs = [self.atk_policy.prob(self._get_atk_ob(t, self.belief, self.last_obs_n[0]), actions[0]) for t in range(self.n_targets)]
        # self.belief = self.update_belief(self.belief, np.array(probs))
        self.belief = self.update_belief(self.belief, probs)
        self.round_cnt += 1

        def_p_bak = np.copy(self.def_p)
        self.def_p[0] += self.v[actions[1]][0]
        self.def_p[1] += self.v[actions[1]][1]
        x, y = self.def_p
        if x < 0 or x >= self.grids.shape[0] / 2 or y < 0 or y >= self.grids.shape[1]:
            self.def_p = np.copy(def_p_bak)

        # print('cur pos:')
        # print(self.att_p, self.def_p)

        att_p_bak = np.copy(self.att_p)
        self.att_p[0] += self.v[actions[0]][0]
        self.att_p[1] += self.v[actions[0]][1]
        x, y = self.att_p
        if x < 0 or x >= self.grids.shape[0] or y < 0 or y >= self.grids.shape[1]:
            self.att_p = np.copy(att_p_bak)

        if np.all(self.att_p == self.def_p):
            self.grids[def_p_bak] = self.grids[att_p_bak] = 0
            self.grids[self.att_p] = 3
        else:
            # print('ready update')
            # print(def_p_bak, att_p_bak)
            # print(self.grids.shape)
            self.grids[def_p_bak] = self.grids[att_p_bak] = 0
            self.grids[self.att_p] = 1
            self.grids[self.def_p] = 2
        
        # rews = self._get_payoff(actions)
        obs_ret = self._get_obs()
        rew_atk = -0.25 * (calc_dis(self.att_p, self.target[self.type]))
        rew_def = -0.25 * (calc_dis(self.att_p, self.def_p))
        if actions[1] == 5:
            self.prob_cnt = self.prob_cnt + 1
            rew_def -= 0.25 * self.prob_cnt
            assert False
        elif actions[1] == 4: # and calc_dis(self.att_p, self.def_p) <= 6.25 ** 0.2 and self.att_p[0] < self.grids.shape[0] / 2:
            # obs_ret[1][-1] = 2
            # print('do tag!!!')
            obs_ret[1][-1] = obs_ret[1][-2] = 0
            if self.type > 0:
                rew_atk = -10
                rew_def = 10
            else:
                rew_def = -20
                rew_atk = -10
        else:
            # obs_ret[1][-1] = 2
            obs_ret[1][-1] = obs_ret[1][-2] = 0
            if actions[1] > 3:
                print('error action!')
                print(actions)
                print(calc_dis(self.att_p, self.def_p))
                print(self.att_p)
        
        rew = (rew_atk, rew_def)

        done = self.round_cnt >= self.n_rounds 
        # or np.all(self.att_p == self.target[self.goal]) or (actions[1] == 4 and calc_dis(self.att_p, self.def_p) <= 6.25 ** 0.2)
        # print(obs_ret.shape)
        self.last_obs_n = obs_ret

        return (obs_ret,
                rew,
                done,
                actions)
    


    def calc_exploitability(self, i, strategy):
        prob = strategy(np.zeros(1))
        j = 1 if i == 0 else 0
        exp = -1e100
        for aj in range(self.na[j]):
            ret = 0.
            for ai in range(self.na[i]):
                a = [None, None]
                a[i] = ai
                a[j] = aj
                tmp = self._get_payoff(a)[j]
                ret += prob[ai] * tmp
            exp = max(exp, ret)
        return exp
