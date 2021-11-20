from typing import Type
import numpy as np
import gym
from gym import spaces
from app import logger as lg
from typing import Type


class Q_learning:
    """
    The Q-learning algorithm as a class
    """

    def __init__(
        self, env: gym.Env, learning_rate: float = 1e-3, discount_factor: float = 0.99
    ):
        self.env = env
        self.q_table = None
        self.check_env(env)
        self.dim = (env.observation_space.n, env.action_space.n)

    def init_q_table(self, dim: tuple = None, init_value: float = 0.0) -> np.array:
        """
        Initialize the q_table
        """
        if not dim:
            dim = self.dim
        q_table = np.ones(dim) * init_value
        return q_table

    def check_env(self, env: gym.Env):
        """
        Check if the environment is in this lib's scope, it must be a discrete gym.Env
        """
        for space, space_name in zip(
            [env.action_space, env.observation_space],
            ["action space", "observation space"],
        ):
            if not isinstance(space, spaces.Discrete):
                raise ValueError(f"{space_name} should be Discrete, not {space}")

    def update(
        self,
        action: int,
        state: int,
        reward: float,
        next_state: int,
        next_action: int = None,
        params: dict = {},
        learning_rate: float = 1e-3,
        discount_factor: float = 0.9,
        verbose=False,
    ) -> float:
        """
        Compute the q table update based on the current state, the previous action, the next state and action
        """
        if "learning_rate" in params.keys():
            learning_rate = params["learning_rate"]
        if "discount_factor" in params.keys():
            discount_factor = params["discount_factor"]
        new_value = self.q_table[state, action] + learning_rate * (
            reward
            + discount_factor * max(self.q_table[next_state])
            - self.q_table[state, action]
        )
        if verbose:
            lg.info(
                f"old q-value {self.q_table[state, action]}, new q_value {new_value}"
            )
        return new_value

    def select_action(
        self,
        q_table: np.array,
        state: int,
        params: dict = {},
        method: str = "epsilon-greedy",
        epsilon: float = 0.99,
    ) -> int:
        """
        Select the action for a specific state and a specific method given the current Q-table
        """
        assert q_table.shape == self.dim
        state_action_values = q_table[state]
        if method == "epsilon-greedy":
            if "epsilon" in params.keys():
                epsilon = params["epsilon"]
            if np.random.rand() > epsilon:
                # Get all actions which have the maximum value for the current state
                max_value_actions = (
                    np.argwhere(state_action_values == np.amax(state_action_values))
                    .flatten()
                    .tolist()
                )
                # Select one at random
                return np.random.choice(max_value_actions)
            else:
                return self.env.action_space.sample()

        elif method == "greedy":
            # Get all actions which have the maximum value for the current state
            max_value_actions = (
                np.argwhere(state_action_values == np.amax(state_action_values))
                .flatten()
                .tolist()
            )
            # Select one at random
            action = np.random.choice(max_value_actions)
            # if len(max_value_actions) > 1:
            #     lg.info(f"{state=}{max_value_actions}{action}")
            return action
        else:
            raise ValueError(f"Method {method} not yet implemented or is non-existent")

    def train(self, params: dict = {}, n_episode: int = 1, verbose=False):
        if not self.q_table:
            self.q_table = self.init_q_table()
        reward_training = []
        for episode in range(n_episode):
            state = self.env.reset()
            done = False
            reward_episode = []
            while not done:
                action = self.select_action(
                    self.q_table, state, params, method="epsilon-greedy"
                )
                next_state, reward, done, info = self.env.step(action)
                if verbose:
                    lg.debug(
                        f"{episode=} : {state=}, {action=}, {next_state=}, {reward=}"
                    )
                self.q_table[state, action] = self.update(
                    action,
                    state,
                    reward,
                    next_state,
                    params=params,
                )
                state = next_state
                reward_episode.append(reward)
            reward_training.append(reward_episode)
        result_dict = {"Training rewards": reward_training}
        self.result_report(result_dict)

    def test(self, n_episode: int = 1, verbose=False) -> dict:
        if self.q_table is None:
            raise ValueError("Test called without a valid q-table")
        reward_training = []
        for episode in range(n_episode):
            state = self.env.reset()
            done = False
            reward_episode = []
            while not done:
                action = self.select_action(self.q_table, state, method="greedy")
                next_state, reward, done, info = self.env.step(action)
                if verbose:
                    lg.info(
                        f"{episode=} : {state=}, {action=}, {next_state=}, {reward=}"
                    )
                state = next_state
                reward_episode.append(reward)
            reward_training.append(reward_episode)
        result_dict = {"Training rewards": reward_training}
        return self.result_report(result_dict)

    def result_report(self, result_dict: dict, verbose: bool = False) -> dict:
        score_threshold = 1
        reward_sum_per_episode = np.array(
            [np.sum(episode) for episode in result_dict["Training rewards"]]
        )
        num_successful_episodes = int(
            np.sum(reward_sum_per_episode[reward_sum_per_episode >= score_threshold])
        )
        if verbose:
            lg.info(
                f"{num_successful_episodes=}/{len(reward_sum_per_episode)} ({num_successful_episodes*100/len(reward_sum_per_episode)})%"
            )
        return {
            "success_rate": num_successful_episodes * 100 / len(reward_sum_per_episode)
        }
