import unittest
from math import hypot

import numpy as np

from toriel_rl.bullets.blue_ring import BlueRingPattern
from toriel_rl.bullets.orange_gates import OrangeGatePattern
from toriel_rl.bullets.rain import RainPattern
from toriel_rl.bullets.sine_sweeper import SineSweeperPattern
from toriel_rl.config import EnvConfig


class SineSweeperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()
        self.rng = np.random.default_rng(1)

    def test_bullets_cross_the_board_with_sine_wobble(self) -> None:
        pattern = SineSweeperPattern(self.config)
        pattern.level = 1
        pattern.reset(self.rng)

        self.assertEqual(len(pattern.bullets()), 5)
        self.assertTrue(all(bullet.color == "white" for bullet in pattern.bullets()))
        start = [(bullet.x, bullet.y) for bullet in pattern.bullets()]
        for _ in range(10):
            pattern.update(self.config.time_step)

        for index, bullet in enumerate(pattern.bullets()):
            self.assertGreater(bullet.x, start[index][0])
        self.assertTrue(
            any(abs(bullet.y - start[index][1]) > 0.5 for index, bullet in enumerate(pattern.bullets()))
        )

    def test_higher_level_spawns_more_and_faster_bullets(self) -> None:
        easy = SineSweeperPattern(self.config)
        easy.level = 1
        easy.reset(self.rng)
        hard = SineSweeperPattern(self.config)
        hard.level = 4
        hard.reset(self.rng)

        self.assertGreater(len(hard.bullets()), len(easy.bullets()))
        self.assertGreater(hard.bullets()[0].vx, easy.bullets()[0].vx)


class BlueRingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()
        self.rng = np.random.default_rng(2)

    def test_ring_contracts_from_edge_toward_center(self) -> None:
        pattern = BlueRingPattern(self.config)
        pattern.level = 1
        pattern.reset(self.rng)
        pattern.update(0.01)
        center_x = self.config.board_width / 2
        center_y = self.config.board_height / 2

        self.assertEqual(len(pattern.bullets()), 12)
        self.assertTrue(all(bullet.color == "blue" for bullet in pattern.bullets()))
        start_radii = [hypot(bullet.x - center_x, bullet.y - center_y) for bullet in pattern.bullets()]
        self.assertTrue(all(abs(radius - pattern.start_radius) < 5.0 for radius in start_radii))

        for _ in range(int((pattern.contract_duration - 0.5) / self.config.time_step)):
            pattern.update(self.config.time_step)
        end_radii = [hypot(bullet.x - center_x, bullet.y - center_y) for bullet in pattern.bullets()]

        self.assertLess(max(end_radii), pattern.start_radius * 0.6)


class OrangeGatesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()
        self.rng = np.random.default_rng(3)

    def test_walls_sweep_right_and_keep_a_passable_gap(self) -> None:
        pattern = OrangeGatePattern(self.config)
        pattern.level = 1
        pattern.reset(self.rng)

        self.assertGreater(len(pattern.bullets()), 0)
        self.assertTrue(all(bullet.color == "orange" for bullet in pattern.bullets()))
        gap_y = pattern._walls[0]["gap_y"]
        self.assertTrue(all(abs(bullet.y - gap_y) > pattern.gap_half for bullet in pattern.bullets()))
        start_x = pattern.bullets()[0].x

        pattern.update(self.config.time_step)

        self.assertAlmostEqual(pattern.bullets()[0].x, start_x + pattern.sweep_speed * self.config.time_step)

    def test_higher_level_uses_more_walls(self) -> None:
        pattern = OrangeGatePattern(self.config)
        pattern.level = 4
        pattern.reset(self.rng)

        self.assertEqual(len(pattern._walls), 3)


class RainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()
        self.rng = np.random.default_rng(4)

    def test_rain_spawns_falling_white_bullets(self) -> None:
        pattern = RainPattern(self.config)
        pattern.level = 2
        pattern.reset(self.rng)

        for _ in range(30):
            pattern.update(self.config.time_step)

        self.assertGreater(len(pattern.bullets()), 0)
        self.assertTrue(all(bullet.color == "white" for bullet in pattern.bullets()))
        self.assertGreater(pattern.bullets()[0].vy, 0.0)

    def test_ramp_mode_increases_spawn_rate_over_time(self) -> None:
        pattern = RainPattern(self.config, ramp=True)
        pattern.level = 1
        pattern.reset(self.rng)
        initial_rate = pattern.spawn_rate

        pattern._elapsed = pattern.ramp_duration

        self.assertGreater(pattern.spawn_rate, initial_rate)
        self.assertAlmostEqual(pattern.spawn_rate, 6.0)


if __name__ == "__main__":
    unittest.main()
