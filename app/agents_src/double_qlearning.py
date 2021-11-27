from logging import warning
from typing import Type
import numpy as np
import gym
from gym import spaces
from app import logger as lg
from typing import Type
from app.agents_src.base import BaseQTable
from tqdm import tqdm


class Double_Q_learning(BaseQTable):
    """
    The Q-learning algorithm as a class
    """

    def __init__(
        self, env: gym.Env, learning_rate: float = 1e-3, discount_factor: float = 0.99
    ):
        super().__init__(env, learning_rate, discount_factor)
        self.name = "Double-Q-learning"
        self.q_table_A = None
        self.q_table_B = None

    def update(
        self,
        action: int,
        state: int,
        reward: float,
        next_state: int,
        q_table_1: np.array,
        q_table_2: np.array,
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
        if "learning_rate" in params.keys():
            learning_rate = params["learning_rate"]
        if "discount_factor" in params.keys():
            discount_factor = params["discount_factor"]
        greedy_action_wrt_1 = np.random.choice(
            np.argwhere(q_table_1[next_state] == np.amax(q_table_1[next_state]))
            .flatten()
            .tolist()
        )
        new_value = q_table_1[state, action] + learning_rate * (
            reward
            + discount_factor * q_table_2[next_state, greedy_action_wrt_1]
            - q_table_1[state, action]
        )
        if verbose:
            lg.info(f"old q-value {q_table_1[state, action]}, new q_value {new_value}")
        return new_value

    def train(self, params: dict = {}, n_episode: int = 1, verbose=False):
        """
        Train the selected agent on the current environment

        Args:
            params (dict, optional): [Training parameters : learning rate, discount_factor, epsilon for epsilon greedy]. Defaults to {}.
            n_episode (int, optional): [Number of epiodes to train on]. Defaults to 1.
            verbose (bool, optional): [Wether to log training info or not]. Defaults to False.
        """
        if not self.q_table_A and not self.q_table_B:
            self.q_table_A = self.init_q_table()
            self.q_table_B = self.init_q_table()
        reward_training = []
        for episode in tqdm(range(n_episode)):
            state = self.env.reset()
            done = False
            reward_episode = []
            while not done:
                merged_q_table = self.q_table_A + self.q_table_B
                action = self.select_action(
                    merged_q_table, state, params, method="epsilon-greedy"
                )
                next_state, reward, done, info = self.env.step(action)
                if verbose:
                    lg.debug(
                        f"{episode=} : {state=}, {action=}, {next_state=}, {reward=}"
                    )
                if np.random.rand() > 0.5:
                    self.q_table_A[state, action] = self.update(
                        action,
                        state,
                        reward,
                        next_state,
                        self.q_table_A,
                        self.q_table_B,
                        params=params,
                    )
                else:
                    self.q_table_B[state, action] = self.update(
                        action,
                        state,
                        reward,
                        next_state,
                        self.q_table_B,
                        self.q_table_A,
                        params=params,
                    )
                state = next_state
                reward_episode.append(reward)
            reward_training.append(reward_episode)
        self.q_table = self.q_table_A + self.q_table_B
        result_dict = {"Training rewards": reward_training}
        self.result_report(result_dict)

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
        if gym.Env is None:
            env = self.env
        if self.q_table_A is None and self.q_table_B is None:
            warning("Q-table is not provided or not valid, had to init a q-table")
            self.q_table_A = self.init_q_table()
            self.q_table_B = self.init_q_table()
        merged_q_table = self.q_table_A + self.q_table_B
        if mode == "test":
            action = self.select_action(merged_q_table, state, method="greedy")
        elif mode == "train":
            action = self.select_action(
                self.q_table, state, params, method="epsilon-greedy"
            )
            next_state, reward, done, info = env.step(action)
            if np.random.rand() > 0.5:
                self.q_table_A[state, action] = self.update(
                    action,
                    state,
                    reward,
                    next_state,
                    self.q_table_A,
                    self.q_table_B,
                    params=params,
                )
            else:
                self.q_table_B[state, action] = self.update(
                    action,
                    state,
                    reward,
                    next_state,
                    self.q_table_B,
                    self.q_table_A,
                    params=params,
                )
        else:
            raise ValueError(f"Mode {mode} not yet implemented or is non-existent")

        return self.env.step(action)
