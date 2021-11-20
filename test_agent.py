import unittest

import gym
import logging
import sys
from app.agents import SARSA
from app.agents import Q_learning
from app import logger as lg
from parameterized import parameterized
import app.config as config

lg.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
lg.addHandler(stream_handler)

env = gym.make("FrozenLake-v1", is_slippery=True)


class TestAgent(unittest.TestCase):
    def test_init_q_table(self):
        Q_Learner = SARSA(env)
        # case 0
        self.assertEqual(
            Q_Learner.init_q_table().shape,
            (env.observation_space.n, env.action_space.n),
        )
        # case 1
        self.assertEqual(
            set(
                Q_Learner.init_q_table(init_value=1).reshape(
                    -1,
                )
            ),
            {1},
        )

    def test_test(self):
        Q_Learner = SARSA(env)
        with self.assertRaises(ValueError):
            Q_Learner.test()
        Q_Learner.train()
        Q_Learner.test()

    @parameterized.expand([[SARSA], [Q_learning]])
    def test_full_train_test(self, agent_class):
        # No slip
        env = gym.make("FrozenLake-v1", is_slippery=False)
        Q_Learner = agent_class(env)
        lg.info(
            f"Testing {Q_Learner.name} on environment {str(env.env.spec).split('(')[1][:-1]}"
        )
        params = {"learning_rate": 0.1, "epsilon": 0.05, "discount factor": 0.999}
        Q_Learner.train(n_episode=1000, params=params, verbose=False)
        self.assertEqual(
            Q_Learner.test(n_episode=1, verbose=False)["success_rate"], 100.0
        )
        # Slippery
        env = gym.make("FrozenLake-v1", is_slippery=True)
        Q_Learner = agent_class(env)
        params = {"learning_rate": 0.1, "epsilon": 0.05, "discount factor": 0.999}
        Q_Learner.train(n_episode=20000, params=params, verbose=False)
        agent_conf = f"{Q_Learner.name} {str(env.env.spec).split('(')[1][:-1]} {params}"
        self.assertGreaterEqual(
            Q_Learner.test(n_episode=2000, verbose=False, agent_conf=agent_conf)[
                "success_rate"
            ],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()