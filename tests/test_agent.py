import tempfile
import unittest
from pathlib import Path

import numpy as np

from toriel_rl.agent import QLearningAgent
from toriel_rl.config import EnvConfig
from toriel_rl.evaluation import evaluate_policies, save_results, summarize
from toriel_rl.training import train_agent


class QLearningAgentTests(unittest.TestCase):
    def test_terminal_update_uses_reward_without_bootstrapping(self) -> None:
        agent = QLearningAgent(learning_rate=1.0, seed=1)
        observation = np.asarray(
            [0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.25, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )

        agent.learn(
            observation,
            4,
            -10.0,
            observation,
            terminated=True,
            truncated=False,
        )

        state = agent.discretize(observation)
        self.assertEqual(agent.q_table[state][4], -10.0)

    def test_model_round_trip_preserves_greedy_action(self) -> None:
        agent = QLearningAgent(seed=2)
        observation = np.asarray(
            [0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.25, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        agent.learn(
            observation,
            3,
            1.0,
            observation,
            terminated=True,
            truncated=False,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "q_table.json"
            agent.save(model_path)
            loaded_agent = QLearningAgent.load(model_path)

        self.assertEqual(loaded_agent.act(observation, explore=False), 3)

    def test_old_model_format_requires_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "old_q_table.json"
            model_path.write_text('{"state_version": 3}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "retrain"):
                QLearningAgent.load(model_path)

    def test_training_saves_model_and_evaluation_metrics(self) -> None:
        config = EnvConfig(episode_seconds=1 / 30)
        with tempfile.TemporaryDirectory() as temporary_directory:
            records = train_agent(
                episodes=3,
                seed=11,
                output_dir=temporary_directory,
                config=config,
                full_combat_episodes=1,
            )
            model = QLearningAgent.load(Path(temporary_directory) / "q_table.json")
            results = evaluate_policies(model, episodes=2, seed=50, config=config)
            evaluation_path = Path(temporary_directory) / "evaluation.csv"
            save_results(results, evaluation_path)

            self.assertTrue((Path(temporary_directory) / "training.csv").exists())
            self.assertTrue(evaluation_path.exists())
            self.assertEqual(len(records), 4)
            self.assertEqual(records[-1]["attack"], "full_combat")
            self.assertEqual(len(results["fire_helix"]["random"]), 2)
            self.assertEqual(len(results["paw_sweep"]["q_learning"]), 2)
            self.assertEqual(len(results["full_combat"]["q_learning"]), 2)
            self.assertIn("mean_damage", summarize(results["fire_waves"]["q_learning"]))


if __name__ == "__main__":
    unittest.main()