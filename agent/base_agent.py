from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def get_initial_policy(self):
        """
        Get the initial policy.
        :return: the initial policy
        """
        pass

    @abstractmethod
    def get_final_policy(self):
        """
        Get the final policy for test.
        :return: the initial policy
        """
        pass

    # @abstractmethod
    # def update(self, trajectory):
    #     """
    #     Update on a trajectory.
    #     :param trajectory: the newest trajectory
    #     :return: the updated policy or None for no update
    #     """
    #     pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_config(self):
        """
        Get the running config of this agent.
        :return: the config
        """
        pass

    def save(self, save_path):
        pass

    def load(self, load_path):
        pass

    def load_sub(self, load_path):
        pass
