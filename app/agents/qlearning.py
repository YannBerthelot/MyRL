from logging import warning
from typing import Type
import numpy as np
import gym
from gym import spaces
from app import logger as lg
from typing import Type
from app.agents.base import BaseQTableMethod


class Q_learning(BaseQTableMethod):
    """
    The Q-learning algorithm as a class
    """

    def __init__(
        self, env: gym.Env, learning_rate: float = 1e-3, discount_factor: float = 0.99
    ):
        super().__init__(env, learning_rate, discount_factor)

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

    def step_model(
        self,
        env: gym.Env,
        state: int,
        q_table: np.array = None,
        mode: str = "test",
        params: dict = {},
    ):
        """
        Performs a single step of test or train and returns obs, reward, info, and done
        """
        if gym.Env is None:
            env = self.env
        if q_table is None:
            warning("Q-table is not provided or not valid, had to init a q-table")
            q_table = self.init_q_table()
        if mode == "test":
            action = self.select_action(q_table, state, method="greedy")
        elif mode == "train":
            action = self.select_action(
                self.q_table, state, params, method="epsilon-greedy"
            )
            next_state, reward, done, info = env.step(action)
            q_table[state, action] = self.update(
                action,
                state,
                reward,
                next_state,
                params=params,
            )
        else:
            raise ValueError(f"Mode {mode} not yet implemented or is non-existent")

        return self.env.step(action)
