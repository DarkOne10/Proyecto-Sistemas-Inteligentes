import unittest

import numpy as np

from toriel_rl.config import EnvConfig
from toriel_rl.entities.fireball import Fireball
from toriel_rl.modes import MODES, create_mode
from toriel_rl.scenarios import SCENARIO_TYPES, create_scenario
from toriel_rl.sandbox_env import SoulSandboxEnv


class StaticPattern:
    """Test helper exposing the pattern protocol with a fixed bullet list."""

    name = "static"
    damage = 4
    level = 1

    def __init__(self, bullets: list[Fireball]) -> None:
        self._bullets = list(bullets)

    def reset(self, rng) -> None:
        return None

    def update(self, dt: float) -> None:
        return None

    def bullets(self) -> list[Fireball]:
        return self._bullets

    def remove(self, bullet: Fireball) -> None:
        self._bullets.remove(bullet)

    def external_displacement(self, dt: float) -> tuple[float, float]:
        return (0.0, 0.0)


class ModeIntegrationTests(unittest.TestCase):
    def test_survival_mode_uses_rain_and_thirty_seconds(self) -> None:
        env = SoulSandboxEnv(mode="survival")
        _, info = env.reset(seed=1)

        self.assertEqual(info["mode"], "survival")
        self.assertEqual(info["pattern"], "rain")
        self.assertEqual(env.steps_per_episode, 900)

    def test_scenario_default_pattern_wins_over_mode_default(self) -> None:
        env = SoulSandboxEnv(scenario="vertical_lanes", mode="survival")
        _, info = env.reset(seed=2)

        self.assertEqual(info["scenario"], "vertical_lanes")
        self.assertEqual(info["pattern"], "empty")

    def test_near_miss_rewards_close_dodges(self) -> None:
        env = SoulSandboxEnv(mode="survival", config=EnvConfig(episode_seconds=10.0))
        env.reset(seed=3, options={"pattern": "empty"})
        soul = env.soul
        env.pattern = StaticPattern(
            [Fireball(x=soul.x + soul.radius + 4.0 + 5.0, y=soul.y, vx=0.0, vy=0.0)]
        )

        _, reward, _, _, _ = env.step(0)

        self.assertAlmostEqual(reward, 0.12)

    def test_unknown_mode_and_scenario_are_rejected(self) -> None:
        env = SoulSandboxEnv()

        with self.assertRaisesRegex(ValueError, "Unknown mode"):
            env.reset(seed=1, options={"mode": "boss_rush"})
        with self.assertRaisesRegex(ValueError, "Unknown scenario"):
            env.reset(seed=1, options={"scenario": "spiral"})


class VerticalLanesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()

    def test_bullets_spawn_outside_the_safe_lane(self) -> None:
        scenario = create_scenario("vertical_lanes", self.config)
        scenario.level = 2
        scenario.reset(np.random.default_rng(5))
        for _ in range(30):
            scenario.update(self.config.time_step)

        lane_width = self.config.board_width / 4
        self.assertGreater(len(scenario.bullets()), 0)
        for bullet in scenario.bullets():
            self.assertNotEqual(int(bullet.x // lane_width), scenario._gap_lane)
            self.assertEqual(bullet.color, "white")

    def test_safe_lane_rotates_on_a_timer(self) -> None:
        scenario = create_scenario("vertical_lanes", self.config)
        scenario.level = 1
        scenario.reset(np.random.default_rng(6))
        initial_gap = scenario._gap_lane

        for _ in range(int(scenario.rotation_interval / self.config.time_step) + 2):
            scenario.update(self.config.time_step)

        self.assertNotEqual(scenario._gap_lane, initial_gap)

    def test_scenario_registers_in_registry(self) -> None:
        self.assertIn("vertical_lanes", SCENARIO_TYPES)
        self.assertIn("survival", MODES)
        self.assertEqual(create_mode("survival", self.config).name, "survival")


if __name__ == "__main__":
    unittest.main()
