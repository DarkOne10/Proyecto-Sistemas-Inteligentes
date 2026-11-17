"""Regression tests: bullets must live on the real Bullet Board of each scenario."""

import unittest

import numpy as np

from toriel_rl.attacks.faltering import FalteringAttack
from toriel_rl.config import EnvConfig
from toriel_rl.sandbox_env import SoulSandboxEnv


def _run(env: SoulSandboxEnv, seconds: float, check) -> None:
    env.reset(seed=3)
    env.soul.hp = 10**6  # never die: we only care about bullet geometry
    for _ in range(round(seconds / env.config.time_step)):
        check(env)
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        if terminated or truncated:
            break


class BoardGeometryTests(unittest.TestCase):
    def test_narrow_corridor_darts_stay_inside_the_corridor(self) -> None:
        env = SoulSandboxEnv(scenario="narrow_corridor", action_mode="continuous",
                             config=EnvConfig(episode_seconds=12.0))

        def check(env):
            self.assertEqual(env.pattern.config.board_height, 64.0)
            for bullet in env.all_bullets():
                self.assertLessEqual(bullet.y, env.board_height + 20.0)

        _run(env, 12.0, check)

    def test_spiral_is_centred_on_the_smaller_board(self) -> None:
        env = SoulSandboxEnv(scenario="spiral_evasion", action_mode="continuous")
        env.reset(seed=0)
        self.assertEqual(
            (env.pattern.config.board_width, env.pattern.config.board_height), (176.0, 176.0)
        )

    def test_collapsing_arena_rain_follows_the_shrinking_board(self) -> None:
        env = SoulSandboxEnv(mode="survival", scenario="collapsing_arena", action_mode="continuous")
        spawned_x: list[float] = []

        def check(env):
            self.assertAlmostEqual(env.pattern.config.board_width, env.board_width)
            for bullet in env.all_bullets():
                if bullet.y < 0:  # freshly spawned this step
                    spawned_x.append((bullet.x, env.board_width))

        _run(env, 20.0, check)
        self.assertTrue(spawned_x)
        for x, width in spawned_x:
            self.assertLessEqual(x, width)

    def test_collapsing_arena_shrink_does_not_compound(self) -> None:
        env = SoulSandboxEnv(scenario="collapsing_arena", action_mode="continuous",
                             config=EnvConfig(episode_seconds=20.0))
        _run(env, 20.0, lambda env: None)
        self.assertAlmostEqual(env.board_width, 192.0 * 0.7, places=3)

    def test_slalom_rows_never_pop_in_mid_board(self) -> None:
        env = SoulSandboxEnv(mode="focus_precision", scenario="slalom_switching", level=3)
        previous: set[int] = set()
        env.reset(seed=1)
        env.soul.hp = 10**6
        for _ in range(round(30.0 / env.config.time_step)):
            current = {id(b): b for b in env.all_bullets()}
            for key, bullet in current.items():
                if previous and key not in previous:
                    self.assertLess(bullet.y, 0.0, "a wall row appeared inside the board")
            previous = set(current)
            _, _, terminated, truncated, _ = env.step(np.array([0, 0]))
            if terminated or truncated:
                break


class FalteringTests(unittest.TestCase):
    def test_faltering_fireballs_cross_the_board_within_one_round(self) -> None:
        config = EnvConfig()
        attack = FalteringAttack(config)
        attack.reset(np.random.default_rng(0))
        lowest = {id(fb): fb.y for fb in attack.fireballs}
        while attack.elapsed < attack.duration:
            attack.update(config.time_step, -1000.0, -1000.0)
            for fb in attack.fireballs:
                lowest[id(fb)] = max(lowest[id(fb)], fb.y)
        reached = sum(1 for y in lowest.values() if y >= config.board_height * 0.9)
        self.assertGreaterEqual(reached, len(lowest) - 1)


class HomingTests(unittest.TestCase):
    def test_homing_bullets_stop_steering_and_leave_the_board(self) -> None:
        from toriel_rl.bullets.homing_sine import HomingSinePattern

        config = EnvConfig()
        pattern = HomingSinePattern(config)
        pattern.reset(np.random.default_rng(0))
        pattern.update(pattern.spawn_interval, 96.0, 96.0)  # spawn exactly one bullet
        self.assertEqual(len(pattern.bullets()), 1)
        pattern._spawn_timer = -1e9  # no more spawns
        bullet = pattern.bullets()[0]
        for _ in range(round(pattern.homing_seconds / config.time_step) + 2):
            pattern.update(config.time_step, 96.0, 96.0)
        heading = (bullet.vx, bullet.vy)
        for _ in range(10):
            pattern.update(config.time_step, 10.0, 180.0)  # SOUL jumps: bullet must not follow
        self.assertAlmostEqual(heading[0], bullet.vx)
        self.assertAlmostEqual(heading[1], bullet.vy)
        for _ in range(round(10.0 / config.time_step)):
            pattern.update(config.time_step, 96.0, 96.0)
        self.assertEqual(pattern.bullets(), [])


class GauntletModeTests(unittest.TestCase):
    def test_gauntlet_rolls_patterns_scenarios_and_levels(self) -> None:
        env = SoulSandboxEnv(mode="gauntlet", action_mode="focus")
        scenarios, patterns, levels = set(), set(), set()
        for seed in range(80):
            _, info = env.reset(seed=seed)
            scenarios.add(info["scenario"])
            patterns.add(info["pattern"])
            levels.add(info["level"])
        self.assertIn(None, scenarios)
        self.assertGreaterEqual(len(scenarios - {None}), 5)
        self.assertGreaterEqual(len(patterns), 10)
        self.assertNotIn("pickups", patterns)
        self.assertEqual(levels, {1, 2, 3})

    def test_explicit_open_board_overrides_the_mode_roll(self) -> None:
        env = SoulSandboxEnv(mode="gauntlet", action_mode="focus")
        for seed in range(20):
            _, info = env.reset(seed=seed, options={"scenario": None, "pattern": "rain"})
            self.assertIsNone(info["scenario"])
            self.assertEqual(info["pattern"], "rain")


if __name__ == "__main__":
    unittest.main()
