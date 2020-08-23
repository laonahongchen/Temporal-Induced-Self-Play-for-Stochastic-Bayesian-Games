from abc import ABC, abstractmethod
from env.base_env import BaseEnv


class BaseController(ABC):
    def __init__(self, env: BaseEnv):
        self.env = env
        self.num_agents = env.num_agents

    @abstractmethod
    def get_push_handler(self, i):
        pass

    @abstractmethod
    def get_pull_handler(self, i):
        pass

    def get_handlers(self, i):
        return self.get_push_handler(i), self.get_pull_handler(i)

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass
