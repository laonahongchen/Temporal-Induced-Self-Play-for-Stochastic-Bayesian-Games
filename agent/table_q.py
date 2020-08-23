import numpy as np
import random

class TableQAgent():
    def __init__(self, obs_space, action_space, alpha=0.1, gamma=0.9):
        # self.table = [0] * actions * length * height # initialize all Q(s,a) to zero
        # self.action_space = actions
        # self.length = length
        # self.height = height
        self.alpha = alpha
        self.gamma = gamma
        self.action_space = action_space
        self.q_table = dict()
        self.num_episode = 0

    def _index(self, state, action):
        """Return the index of Q([x,y], a) in Q_table."""
        # return a * self.height * self.length + x * self.length + y
        # print('in act:')
        # print(state, action)
        if (tuple(state), action) not in self.q_table:
            self.q_table[(tuple(state), action)] = 0
        # else:
        return self.q_table[(tuple(state), action)]
            # return 

    def _epsilon(self):
        # return 0.1
        # version for better convergence:
        # """At the beginning epsilon is 0.2, after 300 episodes decades to 0.05, and eventually go to 0."""
        return 80. / (self.num_episode + 100) + 0.1

    def act(self, state, is_train=True):
        """epsilon-greedy action selection"""
        if is_train and random.random() < self._epsilon():
            return random.randint(0, self.action_space - 1)
        else:
            # print('action space:')
            # print(self.action_space)
            q_values = np.array([self._index(state, a) for a in range(self.action_space)])
            return np.argmax(q_values)
            # return q_values.index(max(q_values))
            

    def max_q(self, state):
        q_values = [self._index(state, a) for a in range(self.action_space)]
        return max(q_values)

    def update(self, a, s0, s1, r, is_terminated):
        # print('in update')
        # print(s0)
        # print(s1)
        idx = self._index(s0, a)
        q_predict = idx
        q_target = r
        if not is_terminated:
            q_target += self.gamma * self.max_q(s1)
        self.q_table[(tuple(s0), a)] += self.alpha * (q_target - q_predict)
        self.num_episode += 1

class AtkQAgent():
    def __init__(self, n_types, obs_space, action_space, alpha=0.1, gamma=0.9):
        self.agents = [TableQAgent(obs_space, action_space, alpha, gamma) for _ in range(n_types)]
        self.n_types = n_types
    
    def act(self, state, is_train=True):
        # print('in act:')
        # print(state, self.n_types)
        # print(state[1 : self.n_types + 1])
        # print(np.argmax(state[1 : self.n_types + 1]))
        # print('atk act'
        # )
        return self.agents[np.argmax(state[1 : self.n_types + 1])].act(state, is_train)
    
    def update(self, a, s0, s1, r, is_terminated):
        return self.agents[np.argmax(s0[1 : self.n_types + 1])].update(a, s0, s1, r, is_terminated)