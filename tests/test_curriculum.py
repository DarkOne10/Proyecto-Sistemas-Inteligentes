import tempfile
import unittest
from pathlib import Path

from toriel_rl.curriculum import Curriculum, CurriculumStage
from toriel_rl.obs import ObsConfig
from toriel_rl.sandbox_env import SoulSandboxEnv
from toriel_rl.sandbox_training import train_curriculum


class CurriculumStructureTests(unittest.TestCase):
    def test_default_five_stages_with_early_masking(self) -> None:
        curriculum = Curriculum()

        self.assertEqual(len(curriculum.stages), 5)
        self.assertEqual(curriculum.current.name, "intro_drifter")
        self.assertEqual(curriculum.env_kwargs["obs_config"].mask_k, 3)
        self.assertEqual(curriculum.stages[1].env_kwargs["obs_config"].mask_k, 3)
        self.assertIsNone(curriculum.stages[2].env_kwargs.get("obs_config"))
        self.assertFalse(curriculum.finished)

        env = SoulSandboxEnv(**curriculum.env_kwargs)
        _, info = env.reset(seed=1)
        self.assertEqual(env.obs_config.fill_limit, 3)
        self.assertEqual(info["pattern"], "drifter")

        while not curriculum.finished:
            curriculum.advance()
        self.assertTrue(curriculum.finished)

    def test_promotion_needs_return_and_completion(self) -> None:
        curriculum = Curriculum()

        self.assertFalse(curriculum.should_promote({"mean_return": -999.0, "completion_rate": 1.0}))
        self.assertFalse(curriculum.should_promote({"mean_return": 0.0, "completion_rate": 0.0}))
        self.assertTrue(curriculum.should_promote({"mean_return": 0.0, "completion_rate": 1.0}))

        curriculum.advance()
        self.assertEqual(curriculum.current.name, "sine_dodge")
        self.assertTrue(curriculum.should_promote({"mean_return": 0.0, "completion_rate": 0.6}))

    def test_final_stage_uses_focus_action_space(self) -> None:
        curriculum = Curriculum()
        for _ in range(4):
            curriculum.advance()

        self.assertEqual(curriculum.current.name, "slalom_focus")
        env = SoulSandboxEnv(**curriculum.env_kwargs)
        _, info = env.reset(seed=1)

        self.assertEqual(env.action_mode, "focus")
        self.assertEqual(env.action_space.nvec.tolist(), [9, 2])
        self.assertEqual(info["scenario"], "slalom_switching")
        self.assertGreater(len(env.all_bullets()), 0)

    def test_empty_curriculum_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Curriculum([])


class CurriculumTrainingTests(unittest.TestCase):
    def test_tiny_curriculum_runs_end_to_end(self) -> None:
        stages = [
            CurriculumStage(
                "tiny_a", 256,
                {"mode": "survival", "pattern": "drifter", "level": 1, "obs_config": ObsConfig(mask_k=3)},
                -1e9, 0.0,
            ),
            CurriculumStage(
                "tiny_b", 256,
                {"mode": "survival", "pattern": "sine_sweeper", "level": 1},
                -1e9, 0.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            history = train_curriculum(
                seed=1,
                n_envs=2,
                output_dir=temporary_directory,
                eval_episodes=1,
                max_rounds=1,
                stages=stages,
            )

            self.assertEqual(len(history), 2)
            self.assertTrue(all(entry["promoted"] for entry in history))
            self.assertTrue((Path(temporary_directory) / "curriculum.csv").exists())
            self.assertTrue((Path(temporary_directory) / "tiny_a" / "sandbox_ppo.zip").exists())
            self.assertTrue((Path(temporary_directory) / "tiny_b" / "sandbox_training.csv").exists())


if __name__ == "__main__":
    unittest.main()
