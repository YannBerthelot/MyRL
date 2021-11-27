import unittest

import gym
import logging
import sys
from app.agents import SARSA, Q_learning, Double_Q_learning, DynaQ
from app import logger as lg
from parameterized import parameterized
import app.config as config

lg.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
lg.addHandler(stream_handler)

env = gym.make("FrozenLake-v1", is_slippery=True)

N_TEST = 10


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

    @parameterized.expand([[DynaQ]])  # , [SARSA], [Q_learning], [Double_Q_learning]])
    def test_full_train_test(self, agent_class):
        # No slip
        env = gym.make("FrozenLake-v1", is_slippery=False)
        Q_Learner = agent_class(env)
        lg.info(
            f"Testing {Q_Learner.name} on environment {str(env.env.spec).split('(')[1][:-1]}"
        )
        lg.info("No slippery")
        params = {"learning_rate": 0.1, "epsilon": 0.05, "discount factor": 0.999}
        Q_Learner.train(n_episode=1000, params=params, verbose=False)
        self.assertEqual(
            Q_Learner.test(n_episode=1, verbose=False)["success_rate"], 100.0
        )
        # Slippery
        lg.info("Slippery")
        is_slippery = True
        env = gym.make("FrozenLake-v1", is_slippery=is_slippery)

        env_name = str(env.env.spec).split("(")[1][:-1]
        for i in range(N_TEST):
            Q_Learner = agent_class(env)
            params = {"learning_rate": 0.05, "epsilon": 0.05, "discount factor": 0.999}
            agent_conf = {
                "agent": Q_Learner.name,
                "env": env_name,
                "env_params": {"slippery": is_slippery},
                "params": params,
                "i": i,
            }
            Q_Learner.train(n_episode=10000, params=params, verbose=False)
            self.assertGreaterEqual(
                Q_Learner.test(
                    n_episode=1000, verbose=False, agent_conf=agent_conf, export=True
                )["success_rate"],
                0.5,
            )


if __name__ == "__main__":
    unittest.main()