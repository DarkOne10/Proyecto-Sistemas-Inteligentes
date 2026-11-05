import importlib.util
import tempfile
import unittest
from pathlib import Path

from toriel_rl.sandbox_training import evaluate, evaluate_random, train_agent

SB3_AVAILABLE = importlib.util.find_spec("stable_baselines3") is not None


@unittest.skipUnless(SB3_AVAILABLE, "stable-baselines3 is not installed")
class SandboxTrainingTests(unittest.TestCase):
    def test_ppo_training_saves_model_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path, rows = train_agent(
                timesteps=512,
                seed=3,
                mode="survival",
                n_envs=2,
                output_dir=temporary_directory,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue((Path(temporary_directory) / "sandbox_training.csv").exists())
            self.assertGreater(len(rows), 0)
            self.assertIn("return", rows[0])
            self.assertEqual(rows[0]["mode"], "survival")

    def test_evaluation_compares_trained_and_random_policies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path, _ = train_agent(
                timesteps=512,
                seed=4,
                pattern="sine_sweeper",
                n_envs=2,
                output_dir=temporary_directory,
            )

            trained = evaluate(model_path, episodes=1, pattern="sine_sweeper")
            random = evaluate_random(episodes=1, pattern="sine_sweeper")

            for stats in (trained, random):
                self.assertIn("mean_return", stats)
                self.assertIn("mean_hp", stats)
                self.assertIn("completion_rate", stats)


if __name__ == "__main__":
    unittest.main()
