from abc import ABC, abstractmethod
import numpy as np


class BaseEnv(ABC):
    def __init__(self, num_agents, observation_spaces, action_spaces):
        assert len(observation_spaces) == num_agents
        assert len(action_spaces) == num_agents

        self.num_agents = num_agents
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.policies = None

    def get_observation_space(self, i):
        return self.observation_spaces[i]

    def get_action_space(self, i):
        return self.action_spaces[i]

    def get_ob_encoders(self):
        def ob_encoder(ob):
            return hash(tuple(ob.astype(int).tolist()))
        return [ob_encoder] * self.num_agents

    def get_ac_encoders(self):
        def ac_encoder(ac):
            return ac
        return [ac_encoder] * self.num_agents

    def get_n_acs(self):
        # print(self.action_spaces[0].)
        return [ac_space.n for ac_space in self.action_spaces]

    def get_ob_namers(self):
        def ob_namer(ob):
            return str(ob)
        return [ob_namer] * self.num_agents

    # def assess_strategy(self, i, strategy):
    #     raise NotImplementedError

    def assess_strategies(self, strategies, debug=False):
        raise NotImplementedError

    def original_num_agents(self):
        return self.num_agents

    def update_policies(self, policies):
        # print(policies)
        self.policies = policies

    @abstractmethod
    def reset(self, verbose=False):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        """
        pass

    @abstractmethod
    def step(self, actions, action_probs):
        """
        Returns (obs, rews, infos, done):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - infos: a sequence of info objects
         - done: whether this episode is over
        """
        pass


class BaseEnvWrapper(BaseEnv):
    def __init__(self, base_env: BaseEnv, num_agents=None, observation_spaces=None, action_spaces=None):
        self.base_env = base_env
        super().__init__(num_agents=num_agents or base_env.num_agents,
                         observation_spaces=observation_spaces or base_env.observation_spaces,
                         action_spaces=action_spaces or base_env.action_spaces)

    @abstractmethod
    def reset(self, debug=False):
        pass

    def step(self, actions, probs):
        # print(self.base_env)
        return self.base_env.step(actions, probs)
