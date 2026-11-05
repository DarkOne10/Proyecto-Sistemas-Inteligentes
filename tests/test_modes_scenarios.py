import unittest

import numpy as np

from toriel_rl.config import EnvConfig
from toriel_rl.entities.fireball import Fireball
from toriel_rl.entities.soul import Soul
from toriel_rl.modes import MODES, StepContext, create_mode
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

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
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
            scenario.update(self.config.time_step, 0.0, 0.0)

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
            scenario.update(self.config.time_step, 0.0, 0.0)

        self.assertNotEqual(scenario._gap_lane, initial_gap)

    def test_scenario_registers_in_registry(self) -> None:
        self.assertIn("vertical_lanes", SCENARIO_TYPES)
        self.assertIn("survival", MODES)
        self.assertEqual(create_mode("survival", self.config).name, "survival")


class CollectorModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()

    def test_collector_sets_quota_and_obstacles(self) -> None:
        mode = create_mode("collector", self.config)
        mode.level = 2
        mode.reset(np.random.default_rng(1))

        self.assertEqual(mode.target, 6)
        self.assertEqual(mode.default_pattern, "pickups")
        self.assertEqual(len(mode.obstacles()), 2)

    def test_collector_completes_when_quota_reached(self) -> None:
        env = SoulSandboxEnv(mode="collector", config=EnvConfig(episode_seconds=30.0))
        env.reset(seed=4, options={"pattern": "empty"})
        env.pattern = StaticPattern(
            [Fireball(x=env.soul.x, y=env.soul.y, vx=0.0, vy=0.0, color="green")]
        )
        env._greens_total = env.mode.target - 1

        _, reward, terminated, truncated, info = env.step(0)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reason"], "objective_met")
        self.assertEqual(info["greens_total"], env.mode.target)
        self.assertEqual(info["objective"], env.mode.target)
        self.assertEqual(len(env.obstacles), 2)
        self.assertAlmostEqual(reward, 26.02)


class FocusPrecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()

    def test_mode_pulsates_board_and_requires_focus(self) -> None:
        mode = create_mode("focus_precision", self.config)

        self.assertEqual(mode.action_mode, "focus")
        self.assertEqual(mode.default_pattern, "blue_rain")
        self.assertEqual(mode.board_size(0.0), (192.0, 192.0))
        self.assertGreater(mode.board_size(1.0)[0], 192.0)

    def test_focus_near_miss_reward(self) -> None:
        mode = create_mode("focus_precision", self.config)
        soul = Soul(x=96, y=96, radius=5, hp=20)

        def context(focus: bool, gap: float) -> StepContext:
            return StepContext(
                elapsed=0.0, soul=soul, bullets=(), damage=0, greens=0, greens_total=0,
                nearest_gap=gap, moving=True, focus=focus, iframes=False,
                whites_only=True, level=1, terminated=False,
            )

        self.assertAlmostEqual(mode.extra_reward(context(True, 5.0)), 0.15)
        self.assertEqual(mode.extra_reward(context(False, 5.0)), 0.0)
        self.assertEqual(mode.extra_reward(context(True, 50.0)), 0.0)

    def test_discrete_env_rejects_focus_mode(self) -> None:
        env = SoulSandboxEnv(config=self.config)

        with self.assertRaisesRegex(ValueError, "requires action_mode"):
            env.reset(seed=1, options={"mode": "focus_precision"})


class SpiralEvasionTests(unittest.TestCase):
    def test_small_arena_defaults_to_spiral(self) -> None:
        config = EnvConfig()
        scenario = create_scenario("spiral_evasion", config)

        self.assertEqual(scenario.base_size, (176.0, 176.0))
        self.assertEqual(scenario.default_pattern, "spiral_barrage")

        env = SoulSandboxEnv(scenario="spiral_evasion", config=config)
        _, info = env.reset(seed=1)

        self.assertEqual(info["pattern"], "spiral_barrage")
        self.assertEqual((env.board_width, env.board_height), (176.0, 176.0))


class SlalomSwitchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()

    def test_arena_and_default_pattern(self) -> None:
        scenario = create_scenario("slalom_switching", self.config)

        self.assertEqual(scenario.base_size, (192.0, 128.0))
        self.assertEqual(scenario.default_pattern, "empty")

    def test_walls_descend_with_alternating_colors_and_gaps(self) -> None:
        scenario = create_scenario("slalom_switching", self.config)
        scenario.level = 1
        scenario.reset(np.random.default_rng(7))
        first_y = scenario._walls[0]["y"]

        self.assertEqual({bullet.color for bullet in scenario.bullets()}, {"orange", "blue"})
        self.assertNotEqual(scenario._walls[0]["gap_side"], scenario._walls[1]["gap_side"])
        self.assertGreater(len(scenario.bullets()), 5)

        scenario.update(self.config.time_step, 0.0, 0.0)
        self.assertGreater(scenario._walls[0]["y"], first_y)

    def test_wall_rows_have_a_passable_gap(self) -> None:
        scenario = create_scenario("slalom_switching", self.config)
        scenario.level = 1
        scenario.reset(np.random.default_rng(8))
        wall = scenario._walls[0]
        gap_center = self.config.board_width * 0.28
        row = [bullet for bullet in scenario.bullets() if bullet.y == wall["y"]]

        self.assertGreater(len(row), 5)
        for bullet in row:
            self.assertGreater(abs(bullet.x - gap_center), scenario.gap_half)
        self.assertGreater(2 * scenario.gap_half, 2 * self.config.soul_radius)

    def test_recycled_wall_flips_gap_and_color(self) -> None:
        scenario = create_scenario("slalom_switching", self.config)
        scenario.level = 1
        scenario.reset(np.random.default_rng(9))
        wall = scenario._walls[0]
        side, color = wall["gap_side"], wall["color"]
        wall["y"] = self.config.board_height + 41.0

        scenario.update(self.config.time_step, 0.0, 0.0)

        self.assertNotEqual(wall["gap_side"], side)
        self.assertNotEqual(wall["color"], color)
        self.assertLess(wall["y"], 0.0)

    def test_slalom_owns_damage_when_pattern_is_empty(self) -> None:
        env = SoulSandboxEnv(scenario="slalom_switching", config=EnvConfig(episode_seconds=30.0))
        env.reset(seed=1)
        env.scenario._walls[0]["y"] = env.soul.y

        _, _, _, _, info = env.step(0)

        self.assertEqual(env.pattern.name, "empty")
        self.assertEqual(info["damage"], 4)
        self.assertEqual(info["damage_by_color"], {"orange": 4})


if __name__ == "__main__":
    unittest.main()
