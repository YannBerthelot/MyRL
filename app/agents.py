import gym
from app import logger as lg
from app.agents_src.qlearning import Q_learning
from app.agents_src.sarsa import SARSA


class Q_learning(Q_learning):
    """
    The Q-learning algorithm as a class

    Args:
        Q_learning (Class): Q-learning
    """

    def __init__(
        self, env: gym.Env, learning_rate: float = 1e-3, discount_factor: float = 0.99
    ):
        super().__init__(env, learning_rate, discount_factor)
        self.name = "Q-learning"


class SARSA(SARSA):
    """
    The SARSA algorithm as a class

    Args:
        SARSA (Class): SARSA
    """

    def __init__(
        self, env: gym.Env, learning_rate: float = 1e-3, discount_factor: float = 0.99
    ):
        super().__init__(env, learning_rate, discount_factor)
        self.name = "SARSA"