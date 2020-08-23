from env.base_env import BaseEnv
from gym import spaces
import numpy as np

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

class GridEnv(BaseEnv):
    def __init__(self, prior, n_steps, type_fruit = 2, obs_size = 1, grid_shape = (5, 5), zero_sum = False):
        # self.zero_sum = True
        # self.payoff = [[1., -1.], [-1., 1.]]

        # self.zero_sum = False
        # self.payoff = [[[-1., -1.], [-3., 0.]], [[0., -3.], [-2., -2.]]]

        self.v = [ [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]

        self.grids = np.zeros(grid_shape)
        self.n_targets = self.type_fruit = type_fruit
        self.zero_sum = zero_sum
        self.obs_size = obs_size
        self.grid_type_n = self.type_fruit + 4
        self.n_rounds = self.n_steps = n_steps - 1
        self.n_agents = 2
        self.prior = prior
        # self.is_atk = is_atk
        self.random_prior = False
        
        # For one cell, the possible result is nothing, fruits, either agents, both agents
        # The last observation for the attacker is its goal
        observation_spaces = [spaces.Box(low=0., high=10., shape=[((self.grids.shape[0]) * (self.grids.shape[1])) * self.grid_type_n + 4 + self.type_fruit * 2]), spaces.Box(low=0., high=10., shape=[((self.grids.shape[0]) * (self.grids.shape[1])) * self.grid_type_n + 4 + self.type_fruit])]

        action_spaces = [spaces.Discrete(4), spaces.Discrete(4)]

        super().__init__(2, observation_spaces, action_spaces)

        self.reset()
    
    def reset_prior(self, prior):
        self.prior = prior
    
    def get_atk_env(self):
        return GridEnv(self.prior, self.n_steps, True)
    
    def update_policy(self, i, policy):
        if i == 0:
            self.atk_policy = policy
        else:
            self.dfd_policy = policy

    def _get_obs(self):
        cur_x, cur_y = self.att_p
        # atk_obs = []
        # for x in range(cur_x - self.obs_size, cur_x + self.obs_size + 1):
        #     for y in range(cur_x - self.obs_size, cur_x + self.obs_size + 1):
        #         if x < 0 or x >= self.grids.shape[0] or y < 0 or y >= self.grids.shape[1]:
        #             atk_obs.append(0)
        #         else:
        #             atk_obs.append(self.grids[x, y])
        # atk_obs = np.array(atk_obs)
        atk_obs = np.copy(self.grids).reshape(-1)
        
        # atk_obs.append(self.goal)
        cur_x, cur_y = self.def_p
        # def_obs = []
        # for x in range(cur_x - self.obs_size, cur_x + self.obs_size + 1):
        #     for y in range(cur_x - self.obs_size, cur_x + self.obs_size + 1):
        #         if x < 0 or x >= self.grids.shape[0] or y < 0 or y >= self.grids.shape[1]:
        #             def_obs.append(0)
        #         else:
        #             def_obs.append(self.grids[x, y])
        # def_obs = np.array(def_obs)
        # # def_obs.append()
        def_obs = np.copy(atk_obs)
        atk_obs = n_one_hot(self.grid_type_n, atk_obs)
        def_obs = n_one_hot(self.grid_type_n, def_obs)
        
        atk_obs = np.concatenate((np.array([self.round_cnt]), self.belief, atk_obs, self.att_p, self.def_p, one_hot(self.type_fruit, self.goal - 1)))
        def_obs = np.concatenate((np.array([self.round_cnt]), self.belief, def_obs, self.att_p, self.def_p))

        return [np.array(atk_obs), np.array(def_obs)]

    def get_type_space(self, i):
        return spaces.Discrete(self.type_fruit)

    def generate_prior(self):
        x = [0.] + sorted(np.random.rand(self.n_targets - 1).tolist()) + [1.]
        prior = np.zeros(self.n_targets)
        for i in range(self.n_targets):
            prior[i] = x[i + 1] - x[i]
        return prior

    def update_belief(self, belief, probs):
        probs = probs.reshape(-1)
        tmp = belief * probs
        if np.sum(tmp) < 1e-2:
            return np.ones(self.n_targets) / self.n_targets
        return tmp / np.sum(tmp)

    def reset(self, debug=False):
        if self.random_prior:
            self.prior = self.generate_prior()
        # if self.is_atk:
            # self.att_goal = self.goal = self.type = np.random.choice(self.n_targets) + 1
        # else:
        self.att_goal = self.goal = np.random.choice(self.n_targets, p=self.prior) + 1
        self.type = self.goal - 1
        self.round_cnt = 0
        self.belief = np.copy(self.prior)

        self.att_p = np.array([np.random.randint(self.grids.shape[0]), np.random.randint(self.grids.shape[1])])
        self.def_p = np.array([np.random.randint(self.grids.shape[0]), np.random.randint(self.grids.shape[1])])
        while (self.att_p == self.def_p).all():
            self.def_p = [np.random.randint(self.grids.shape[0]), np.random.randint(self.grids.shape[1])]
        
        self.last_obs_n = self._get_obs()
        self.debug = debug

        return self.last_obs_n, None, None
    
    def sub_reset(self, round, belief, debug=False, verbose=False):
        # if self.random_prior:
            # self.prior = self.generate_prior()
        # if self.is_atk:
            # self.att_goal = self.goal = self.type = np.random.choice(self.n_targets) + 1
        # else:
        self.att_goal = self.goal = np.random.choice(self.n_targets, p=belief) + 1
        self.type = self.goal - 1
        self.round_cnt = round
        self.belief = np.copy(belief)

        self.att_p = np.array([np.random.randint(self.grids.shape[0]), np.random.randint(self.grids.shape[1])])
        self.def_p = np.array([np.random.randint(self.grids.shape[0]), np.random.randint(self.grids.shape[1])])
        while (self.att_p == self.def_p).all():
            self.def_p = [np.random.randint(self.grids.shape[0]), np.random.randint(self.grids.shape[1])]
        self.grids = np.random.randint(0, self.n_targets + 1, self.grids.shape)
        self.grids[self.att_p[0], self.att_p[1]] = 0
        self.grids[self.def_p[0], self.def_p[1]] = 0
        # print(self.att_p, self.def_p)
        # print(self.grids)
        
        self.last_obs_n = self._get_obs()
        self.debug = debug

        return self.last_obs_n, None, None

    def step(self, actions, probs):
        # probs = [self.atk_policy.prob(self._get_atk_ob(t, self.belief, self.last_obs_n[0]), actions[0]) for t in range(self.n_targets)]
        self.belief = self.update_belief(self.belief, np.array(probs))
        self.round_cnt += 1

        def_p_bak = np.copy(self.def_p)
        self.def_p[0] += self.v[actions[1]][0]
        self.def_p[1] += self.v[actions[1]][1]
        x, y = self.def_p
        if x < 0 or x >= self.grids.shape[0] or y < 0 or y >= self.grids.shape[1]:
            self.def_p = np.copy(def_p_bak)
            def_r = 0
        else:
            if self.grids[x, y] > 0 and self.grids[x, y] <= self.type_fruit:
                def_r = 1 if self.grids[x, y] == 1 else -1
                self.grids[x, y] = 0
            else:
                def_r = 0

        att_p_bak = np.copy(self.att_p)
        self.att_p[0] += self.v[actions[0]][0]
        self.att_p[1] += self.v[actions[1]][0]
        x, y = self.att_p
        if x < 0 or x >= self.grids.shape[0] or y < 0 or y >= self.grids.shape[1]:
            self.att_p = np.copy(att_p_bak)
            att_r = 0
        else:
            if self.grids[x, y] > 0 and self.grids[x, y] <= self.type_fruit:
                att_r = 1 if self.grids[x, y] == self.att_goal else -1
                self.grids[x, y] = 0
            else:
                att_r = 0

        if (self.att_p == self.def_p).all():
            self.grids[def_p_bak[0], def_p_bak[1]] = self.grids[att_p_bak[0], att_p_bak[1]] = 0
            self.grids[self.att_p[0], self.att_p[1]] = 0
        else:
            self.grids[def_p_bak[0], def_p_bak[1]] = self.grids[att_p_bak[0], att_p_bak[1]] = 0
            self.grids[self.att_p[0], self.att_p[1]] = 0
            self.grids[self.def_p[0], self.def_p[1]] = 0
        
        # print('cur state :')
        # print(self.grids, self.att_p, self.def_p)
        
        if self.zero_sum:
            rew = [att_r - def_r, def_r - att_r]
        else:
            rew = [att_r, def_r]
        # rews = self._get_payoff(actions)
        obs_ret = self._get_obs()
        self.last_obs_n = obs_ret

        return (self.last_obs_n,
                rew,
                self.round_cnt >= self.n_steps, 
                actions)

    def simulate(self, strategies, verbose=False, save_dir=None, prior=None, benchmark=False):
        rp = self.random_prior
        if prior is not None:
            self.prior = np.copy(prior)
            self.random_prior = False
        atk_policy, dfd_policy = strategies
        atk_strategy, dfd_strategy = atk_policy.strategy_fn, dfd_policy.strategy_fn
        ob, _, _ = self.reset(benchmark)
        atk_rew = 0.
        dfd_rew = 0.
        # atk_touching = [[0 for _ in range(self.n_targets)] for _ in range(self.n_rounds)]
        # dfd_touching = [[0 for _ in range(self.n_targets)] for _ in range(self.n_rounds)]
        if verbose:
            print("\nSimulation starts.")
            # self.env.render()
        frames = list()
        if verbose:
            frames.append(self.env.render('rgb_array')[0])
        steps = 0
        sub_steps = 0
        while True:
            atk_s = atk_strategy(ob[0])
            dfd_s = dfd_strategy(ob[1])
            # print(atk_s.shape)
            # print(dfd_s.shape)
            atk_a = np.random.choice(atk_s.shape[0], p=atk_s)
            dfd_a = np.random.choice(dfd_s.shape[0], p=dfd_s)
            if verbose:
                print("Round {} Step {}".format(steps, sub_steps))
                print("Attacker: {} -> {} -> {}".format(ob[0], atk_s, atk_a))
                print("Defender: {} -> {} -> {}".format(ob[1], dfd_s, dfd_a))
                # if sub_steps == 0:
                #     print("tpred:", atk_policy.tpred(ob[0]))
                #     print("tpred:", dfd_policy.tpred(ob[1]))
            ob, rew, _, done, _, _ = self.step([atk_a, dfd_a], None)
            
            # if verbose:
            #     frames.append(self.env.render('rgb_array')[0])
            # self.env.render()

            atk_rew += rew[0]
            dfd_rew += rew[1]
            if done:
                break

        self.random_prior = rp
        return atk_rew, dfd_rew, self.goal

    def _assess_strategies(self, strategies, trials=100, debug=False, prior=None):
        atk_rew = [0., 0.]
        dfd_rew = [0., 0.]
        atk_right = [[0, 0] for _ in range(self.n_rounds)]
        dfd_right = [[0, 0] for _ in range(self.n_rounds)]
        atk_same = 0
        dfd_same = 0
        trials_cnt = [0, 0]
        for _ in range(trials):
            a, d, atk_type = self.simulate(strategies, verbose=False, prior=prior)
            atk_rew[atk_type] += a
            dfd_rew[atk_type] += d
            trials_cnt[atk_type] += 1

        # print("Attacker rights:", np.array(atk_right) / trials, [atk_right[i][0] / atk_right[i][1] for i in range(self.n_rounds)])
        # print("Defender rights:", np.array(dfd_right) / trials, [dfd_right[i][0] / dfd_right[i][1] for i in range(self.n_rounds)])
        # print("Attacker same:", atk_same / trials)
        # print("Defender same:", dfd_same / trials)
        if trials_cnt[0] > 0:
            print("Attacker reward (Defender Align):", atk_rew[0] / trials_cnt[0])
            print("Defender reward (Defender Align):", dfd_rew[0] / trials_cnt[0])
        if trials_cnt[1] > 0:
            print("Attacker reward (Attacker Align):", atk_rew[1] / trials_cnt[1])
            print("Defender reward (Attacker Align):", dfd_rew[1] / trials_cnt[1])

        return [(atk_rew[1] + atk_rew[0]) / trials, (dfd_rew[0] + dfd_rew[1]) / trials]

    def assess_strategies(self, strategies, trials=100, debug=False):
        if self.random_prior:
            for p in range(11):
                prior = np.array([p / 10, 1 - p / 10])
                print("Prior:", prior)
                self._assess_strategies(strategies, trials, debug, prior)
                print("")
        else:
            return self._assess_strategies(strategies, trials, debug)
