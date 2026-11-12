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
            self.assertIn("dodged", rows[0])
            self.assertIn("wall_hits", rows[0])
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

    def test_sac_training_runs_on_continuous_env(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path, rows = train_agent(
                timesteps=256,
                seed=5,
                algo="sac",
                action_mode="continuous",
                pattern="drifter",
                n_envs=2,
                output_dir=temporary_directory,
            )

            self.assertEqual(model_path.name, "sandbox_sac.zip")
            self.assertTrue(model_path.exists())
            self.assertGreater(len(rows), 0)
            self.assertEqual(rows[0]["pattern"], "drifter")

            stats = evaluate(
                model_path,
                episodes=1,
                algo="sac",
                action_mode="continuous",
                pattern="drifter",
            )
            self.assertIn("mean_return", stats)

    def test_sac_rejects_discrete_env(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "continuous"):
                train_agent(
                    timesteps=64,
                    seed=6,
                    algo="sac",
                    pattern="drifter",
                    n_envs=1,
                    output_dir=temporary_directory,
                )


if __name__ == "__main__":
    unittest.main()
