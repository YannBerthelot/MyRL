from logging import warning
from typing import Type
import numpy as np
import gym
from gym import spaces
from app import logger as lg
from typing import Type
import app.config as config


class BaseQTableMethod:
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

        Args:
            action (int): [The agent's action]
            state (int): [The previous state of the environment]
            reward (float): [The reward given by the environment]
            next_state (int): [The new state after the agent's action has been considered]
            next_action (int, optional): [The action taken by the agent given the next step, required in SARSA, unused in Q-learning]. Defaults to None.
            params (dict, optional): [Training parameters for the update : learning rate, discount factor, epsilon for epsilon greedy]. Defaults to {}.
            learning_rate (float, optional): [Learning rate for this update, will be overwritten by params if it includes learning rate]. Defaults to 1e-3.
            discount_factor (float, optional): [Discount factor for this update, will be overwritten by params if it includes discount factor]. Defaults to 0.9.
            verbose (bool, optional): [Wether or not to log training info]. Defaults to False.

        Returns:
            float: [description]
        """
        raise NotImplementedError

    def select_action(
        self,
        q_table: np.array,
        state: int,
        params: dict = {},
        method: str = "epsilon-greedy",
        epsilon: float = 0.99,
    ) -> int:
        """
        Selects the next action given the current Q-table and method

        Args:
            q_table (np.array): [Current version of the Q-Table]
            state (int): [The state the agent is in]
            params (dict, optional): [Action selection params : epsilon for epsilon-greedy]. Defaults to {}.
            method (str, optional): [Action selection methods among "greedy" and "epsilon-greedy"]. Defaults to "epsilon-greedy".
            epsilon (float, optional): [Epsilon parameter for epsilon-greedy method]. Defaults to 0.99.

        Raises:
            ValueError: [Raises an error if the method is invalid]
            NotImplementedError: [Raises an error if the method is not yet implemented]


        Returns:
            int: [The action selected]
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
        elif method == "Boltzmann":
            raise NotImplementedError
        else:
            raise ValueError(f"Method {method} not yet implemented or is non-existent")

    def train(self, params: dict = {}, n_episode: int = 1, verbose=False):
        """
        Train the selected agent on the current environment

        Args:
            params (dict, optional): [Training parameters : learning rate, discount_factor, epsilon for epsilon greedy]. Defaults to {}.
            n_episode (int, optional): [Number of epiodes to train on]. Defaults to 1.
            verbose (bool, optional): [Wether to log training info or not]. Defaults to False.

        Raises:
            NotImplementedError: [Error if the train function is not customized in the children class]
        """
        raise NotImplementedError

    def test(self, n_episode: int = 1, verbose=False, agent_conf: str = "") -> dict:
        """
        Test the agent's performance

        Args:
            n_episode (int, optional): [Number of episodes to train the agent on]. Defaults to 1.
            verbose (bool, optional): [Wether or not to log testing info or not]. Defaults to False.

        Raises:
            ValueError: [Raise an error if there is no valid q-table]

        Returns:
            dict: [Returns the performance report as a dictionnary]
        """
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
        return self.result_report(
            result_dict, config.SHOW_REPORT, agent_conf=agent_conf
        )

    def result_report(
        self, result_dict: dict, verbose: bool = False, agent_conf: str = ""
    ) -> dict:
        """
        Outputs a report on the results to measure agent's performance

        Args:
            result_dict (dict): [Dictionnary containing KPI's on the agent's performance]
            verbose (bool, optional): [Wether or not to print performance during training]. Defaults to False.

        Returns:
            dict: [description]
        """
        score_threshold = 1
        reward_sum_per_episode = np.array(
            [np.sum(episode) for episode in result_dict["Training rewards"]]
        )
        num_successful_episodes = int(
            np.sum(reward_sum_per_episode[reward_sum_per_episode >= score_threshold])
        )
        if verbose:
            lg.info(
                f"{agent_conf} {num_successful_episodes=}/{len(reward_sum_per_episode)} ({num_successful_episodes*100/len(reward_sum_per_episode)})%"
            )
        return {
            "success_rate": num_successful_episodes * 100 / len(reward_sum_per_episode)
        }

    def step_model(
        self,
        env: gym.Env,
        state: int,
        q_table: np.array = None,
        mode: str = "test",
        params: dict = {},
    ):
        """
        Perform a single step of train or test of the agent

        Args:
            env (gym.Env): [The gym environment to make a step on, if None is provided it will take the current defined gym env]
            state (int): [The current state of the environment]
            action (int, optional): [The previous action, for SARSA]. Defaults to None.
            q_table (np.array, optional): [The current Q-table, if None is provided it will be initialized]. Defaults to None.
            mode (str, optional): [Wether to test or train]. Defaults to "test".
            params (dict, optional): [Training parameters : learning rate, discount_factor, epsilon for epsilon greedy]. Defaults to {}.

        Raises:
            ValueError: [Raises error for unimplemented or incorrect modes]

        Returns:
            dict: [Returns a dictionnary containing : next_state, reward, done, info]
        """
        raise NotImplementedError