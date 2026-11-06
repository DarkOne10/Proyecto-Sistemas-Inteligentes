import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from toriel_rl.sandbox_benchmark import DEFAULT_TASKS, benchmark, plot_curves

SB3_AVAILABLE = importlib.util.find_spec("stable_baselines3") is not None
MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None


class BenchmarkStructureTests(unittest.TestCase):
    def test_default_tasks_cover_modes_and_scenarios(self) -> None:
        self.assertIn("survival_rain", DEFAULT_TASKS)
        self.assertIn("slalom_focus", DEFAULT_TASKS)
        self.assertEqual(
            DEFAULT_TASKS["slalom_focus"], {"mode": "focus_precision", "scenario": "slalom_switching"}
        )

    def test_non_positive_timesteps_rejected(self) -> None:
        with self.assertRaises(ValueError):
            benchmark(tasks={}, timesteps=0)

    @unittest.skipUnless(MATPLOTLIB_AVAILABLE, "matplotlib is not installed")
    def test_plot_writes_png_from_curve_rows(self) -> None:
        rows = [
            {"task": "tiny", "algo": "random", "timesteps": 0, "mean_return": -10.0, "mean_hp": 10.0, "completion_rate": 0.0, "seed": 7},
            {"task": "tiny", "algo": "ppo", "timesteps": 256, "mean_return": 1.0, "mean_hp": 18.0, "completion_rate": 0.8, "seed": 7},
            {"task": "tiny", "algo": "ppo", "timesteps": 512, "mean_return": 5.0, "mean_hp": 20.0, "completion_rate": 1.0, "seed": 7},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = plot_curves(rows, Path(temporary_directory) / "curves.png")

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


@unittest.skipUnless(SB3_AVAILABLE, "stable-baselines3 is not installed")
class BenchmarkRunTests(unittest.TestCase):
    def test_tiny_benchmark_writes_curve_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            rows = benchmark(
                tasks={"tiny": {"mode": "survival", "pattern": "drifter"}},
                algos=("ppo",),
                timesteps=256,
                eval_every=256,
                eval_episodes=1,
                n_envs=1,
                output_dir=temporary_directory,
            )

            self.assertEqual(rows[0]["algo"], "random")
            self.assertEqual(rows[1]["algo"], "ppo")
            self.assertGreaterEqual(rows[1]["timesteps"], 256)
            self.assertEqual(rows[1]["task"], "tiny")

            csv_path = Path(temporary_directory) / "benchmark_curves.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open(encoding="utf-8") as csv_file:
                data = list(csv.DictReader(csv_file))
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["task"], "tiny")
            self.assertEqual(int(data[1]["timesteps"]), rows[1]["timesteps"])
            self.assertIn("mean_return", data[1])

    def test_focus_task_skips_sac(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            rows = benchmark(
                tasks={"focus": {"mode": "focus_precision"}},
                algos=("sac",),
                timesteps=64,
                eval_every=64,
                eval_episodes=1,
                n_envs=1,
                output_dir=temporary_directory,
            )

            self.assertEqual([row["algo"] for row in rows], ["random"])


if __name__ == "__main__":
    unittest.main()
