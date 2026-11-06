import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from toriel_rl.agent import QLearningAgent
from toriel_rl.arena import run_arena
from toriel_rl.bullets import PATTERN_TYPES
from toriel_rl.environment import TorielEnv
from toriel_rl.scenarios import SCENARIO_TYPES
from toriel_rl.showcase import (
    SCENE_TEXT,
    ModelPolicy,
    QTablePolicy,
    RandomPolicy,
    Scene,
    build_scenes,
    detect_algo,
    draw_arena_hud,
    draw_title_card,
    load_policies,
    run_showcase,
    scene_options,
)

SB3_AVAILABLE = importlib.util.find_spec("stable_baselines3") is not None

EXPECTED_KEYS = [
    "fire_helix",
    "fire_waves",
    "paw_sweep",
    "sine_sweeper",
    "blue_ring",
    "orange_gates",
    "rain",
    "homing_sine",
    "dart_volley",
    "spiral_barrage",
    "drifter",
    "pickups",
    "blue_rain",
    "cross_fire",
    "ricochet",
    "aimed_burst",
    "zigzag_rain",
    "vortex",
    "vertical_lanes",
    "spiral_evasion",
    "slalom_switching",
    "pinball",
    "narrow_corridor",
    "collapsing_arena",
]


class SceneCatalogTests(unittest.TestCase):
    def test_fixed_scene_order(self) -> None:
        self.assertEqual([scene.key for scene in build_scenes()], EXPECTED_KEYS)

    def test_scene_kinds_and_text(self) -> None:
        scenes = build_scenes()
        self.assertEqual([scene.kind for scene in scenes[:3]], ["attack"] * 3)
        self.assertEqual({scene.kind for scene in scenes[3:18]}, {"pattern"})
        self.assertEqual([scene.kind for scene in scenes[18:]], ["scenario"] * 6)
        for scene in scenes:
            self.assertIn(scene.key, SCENE_TEXT)
            self.assertTrue(scene.title.strip())
            self.assertTrue(scene.rule.strip())

    def test_all_patterns_and_scenarios_are_covered(self) -> None:
        keys = {scene.key for scene in build_scenes()}
        self.assertEqual(set(PATTERN_TYPES) - {"empty"}, keys & set(PATTERN_TYPES))
        self.assertEqual(set(SCENARIO_TYPES), keys & set(SCENARIO_TYPES))
        self.assertNotIn("empty", keys)

    def test_scene_options_select_the_right_family(self) -> None:
        attack = Scene("attack", "fire_helix", "Fire Helix", "rule")
        pattern = Scene("pattern", "rain", "Rain", "rule")
        scenario = Scene("scenario", "pinball", "D4 Pinball", "rule")
        self.assertEqual(scene_options(attack), {"attack": "fire_helix", "attack_only": True})
        self.assertEqual(scene_options(pattern), {"pattern": "rain"})
        self.assertEqual(scene_options(scenario), {"scenario": "pinball"})


class PolicyTests(unittest.TestCase):
    def test_default_policies_are_random(self) -> None:
        policies = load_policies()
        self.assertIsInstance(policies.attack, RandomPolicy)
        self.assertIsInstance(policies.sandbox, RandomPolicy)
        self.assertIsNone(policies.action_mode)
        self.assertEqual(policies.label, "random")

    def test_q_table_runs_only_the_toriel_attacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q_table.json"
            QLearningAgent(seed=3).save(path)
            policies = load_policies(path)

        self.assertIsInstance(policies.attack, QTablePolicy)
        self.assertIsInstance(policies.sandbox, RandomPolicy)
        self.assertIsNone(policies.action_mode)

        env = TorielEnv()
        observation, _ = env.reset(seed=1)
        action = policies.attack.act(observation, env.action_space)
        self.assertIn(action, range(5))
        env.close()

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            run_showcase(seconds=0.0)
        with self.assertRaises(ValueError):
            run_showcase(title_seconds=-1.0)
        with self.assertRaises(ValueError):
            run_arena(episodes=0)


@unittest.skipUnless(SB3_AVAILABLE, "stable-baselines3 is not installed")
class ModelPolicyTests(unittest.TestCase):
    def _save_model(self, algo: str, path: Path) -> None:
        from stable_baselines3.common.env_util import make_vec_env

        from toriel_rl.sandbox_env import SoulSandboxEnv
        from toriel_rl.sandbox_training import make_model

        action_mode = "continuous" if algo == "sac" else None
        vec_env = make_vec_env(
            lambda: SoulSandboxEnv(pattern="drifter", action_mode=action_mode),
            n_envs=1,
            seed=0,
        )
        try:
            make_model(algo, vec_env, 0).save(path)
        finally:
            vec_env.close()

    def test_ppo_zip_drives_the_sandbox_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.zip"
            self._save_model("ppo", path)

            self.assertEqual(detect_algo(path), "ppo")
            policies = load_policies(path)

        self.assertIsInstance(policies.sandbox, ModelPolicy)
        self.assertIsInstance(policies.attack, RandomPolicy)
        self.assertEqual(policies.action_mode, "discrete")
        self.assertEqual(policies.sandbox_obs, (98,))

    def test_sac_zip_detected_as_continuous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.zip"
            self._save_model("sac", path)

            self.assertEqual(detect_algo(path), "sac")
            policies = load_policies(path)

        self.assertEqual(policies.action_mode, "continuous")
        self.assertIsInstance(policies.sandbox, ModelPolicy)


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import pygame

        pygame.init()
        cls.pygame = pygame
        cls.surface = pygame.display.set_mode((576, 608))

    def test_title_card_dims_the_frame(self) -> None:
        self.surface.fill((12, 14, 20))
        before = self.surface.get_at((4, 4))
        draw_title_card(self.surface, build_scenes()[0], self.pygame)
        self.assertNotEqual(self.surface.get_at((4, 4)), before)

    def test_hud_fills_the_bottom_bar(self) -> None:
        from toriel_rl.sandbox_env import SoulSandboxEnv

        env = SoulSandboxEnv()
        _, info = env.reset(seed=1)
        self.surface.fill((12, 14, 20))
        draw_arena_hud(self.surface, env, build_scenes()[3], info, 12.5, self.pygame)

        self.assertEqual(tuple(self.surface.get_at((400, 608 - 16)))[:3], (16, 18, 26))
        env.close()


class ShowcaseRunTests(unittest.TestCase):
    def test_cycles_every_scene_with_title_cards(self) -> None:
        shown = run_showcase(seconds=0.05, title_seconds=0.05, seed=11)
        self.assertEqual(list(shown), EXPECTED_KEYS)

    def test_scene_subset_runs_only_its_scenes(self) -> None:
        scenes = [scene for scene in build_scenes() if scene.kind == "scenario"][:2]
        shown = run_showcase(seconds=0.05, title_seconds=0.0, seed=2, scenes=scenes)
        self.assertEqual([scene.key for scene in scenes], list(shown))


class ArenaRunTests(unittest.TestCase):
    def test_attack_episode(self) -> None:
        scenes = [scene for scene in build_scenes() if scene.kind == "attack"]
        records = run_arena(
            random=True, scenes=scenes, episodes=1, seconds=0.05, title_seconds=0.0, seed=5
        )

        self.assertEqual(len(records), 1)
        self.assertIn(records[0]["scene"], [scene.key for scene in scenes])
        self.assertIsInstance(records[0]["return"], float)
        self.assertIn(records[0]["reason"], {"death", "attack_completed"})

    def test_sandbox_episodes(self) -> None:
        scenes = [scene for scene in build_scenes() if scene.kind != "attack"]
        records = run_arena(
            random=True, scenes=scenes, episodes=2, seconds=0.05, title_seconds=0.0, seed=5
        )

        keys = {scene.key for scene in scenes}
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertIn(record["scene"], keys)
            self.assertIn(record["reason"], {"time_up", "death"})
            self.assertGreaterEqual(record["hp"], 0)


if __name__ == "__main__":
    unittest.main()
