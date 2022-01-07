from logging import warning
from typing import Type
import numpy as np
import gym
from gym import spaces
from app import logger as lg
from typing import Type
from app.agents_src.base import BaseQTable
from tqdm import tqdm


class SARSA(BaseQTable):
    """
    The Q-learning algorithm as a class

    Args:
        BaseQTableMethod (Class): Base Class for Q-table methods
    """

    def __init__(
        self, env: gym.Env, learning_rate: float = 1e-3, discount_factor: float = 0.99
    ):
        super().__init__(env, learning_rate, discount_factor)
        self.name = "SARSA"

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
        if "learning_rate" in params.keys():
            learning_rate = params["learning_rate"]
        if "discount_factor" in params.keys():
            discount_factor = params["discount_factor"]
        new_value = self.q_table[state, action] + learning_rate * (
            reward
            + discount_factor * self.q_table[next_state, next_action]
            - self.q_table[state, action]
        )
        if verbose:
            lg.info(
                f"old q-value {self.q_table[state, action]}, new q-value {new_value}"
            )
        return new_value

    def train(self, params: dict = {}, n_episode: int = 1, verbose=False):
        """
        Train the selected agent on the current environment

        Args:
            params (dict, optional): [Training parameters : learning rate, discount_factor, epsilon for epsilon greedy]. Defaults to {}.
            n_episode (int, optional): [Number of epiodes to train on]. Defaults to 1.
            verbose (bool, optional): [Wether to log training info or not]. Defaults to False.
        """
        if not self.q_table:
            self.q_table = self.init_q_table()
        reward_training = []
        for episode in tqdm(range(n_episode)):
            state = self.env.reset()
            done = False
            reward_episode = []
            action = self.select_action(
                self.q_table, state, params, method="epsilon-greedy"
            )
            while not done:
                next_state, reward, done, info = self.env.step(action)
                next_action = self.select_action(
                    self.q_table, next_state, params, method="epsilon-greedy"
                )
                if verbose:
                    lg.debug(
                        f"{episode=} : {state=}, {action=}, {next_state=}, {reward=}"
                    )
                self.q_table[state, action] = self.update(
                    action,
                    state,
                    reward,
                    next_state,
                    next_action,
                    params=params,
                )
                state = next_state
                action = next_action
                reward_episode.append(reward)
            reward_training.append(reward_episode)
        result_dict = {"Training rewards": reward_training}
        self.result_report(result_dict)

    def step_model(
        self,
        env: gym.Env,
        state: int,
        action: int = None,
        q_table: np.array = None,
        mode: str = "test",
        params: dict = {},
    ) -> dict:
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
            dict: [Returns a dictionnary containing : next_state, reward, done, info and next_action]
        """
        if gym.Env is None:
            env = self.env
        if q_table is None:
            warning("Q-table is not provided or not valid, had to init a q-table")
            q_table = self.init_q_table()

        if mode == "test":
            action = self.select_action(q_table, state, method="greedy")
            next_state, reward, done, info = self.env.step(action)
            return {
                "next_state": next_state,
                "reward": reward,
                "done": done,
                "info": info,
            }
        elif mode == "train":
            if action is None:
                action = self.select_action(
                    self.q_table, state, params, method="epsilon-greedy"
                )
            next_state, reward, done, info = env.step(action)
            next_action = self.select_action(
                self.q_table, next_state, params, method="epsilon-greedy"
            )

            q_table[state, action] = self.update(
                action,
                state,
                reward,
                next_state,
                params=params,
            )
            return {
                "next_state": next_state,
                "reward": reward,
                "done": done,
                "info": info,
                "action": next_action,
            }
        else:
            raise ValueError(f"Mode {mode} not yet implemented or is non-existent")
