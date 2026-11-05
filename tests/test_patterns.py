import unittest
from math import atan2, hypot, tau

import numpy as np

from toriel_rl.bullets.blue_ring import BlueRingPattern
from toriel_rl.bullets.blue_rain import BlueRainPattern
from toriel_rl.bullets.dart_volley import DartVolleyPattern
from toriel_rl.bullets.drifter import DrifterPattern
from toriel_rl.bullets.homing_sine import HomingSinePattern
from toriel_rl.bullets.orange_gates import OrangeGatePattern
from toriel_rl.bullets.pickups import PickupsPattern
from toriel_rl.bullets.rain import RainPattern
from toriel_rl.bullets.sine_sweeper import SineSweeperPattern
from toriel_rl.bullets.spiral_barrage import SpiralBarragePattern
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
            pattern.update(self.config.time_step, 0.0, 0.0)

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
        pattern.update(0.01, 0.0, 0.0)
        center_x = self.config.board_width / 2
        center_y = self.config.board_height / 2

        self.assertEqual(len(pattern.bullets()), 12)
        self.assertTrue(all(bullet.color == "blue" for bullet in pattern.bullets()))
        start_radii = [hypot(bullet.x - center_x, bullet.y - center_y) for bullet in pattern.bullets()]
        self.assertTrue(all(abs(radius - pattern.start_radius) < 5.0 for radius in start_radii))

        for _ in range(int((pattern.contract_duration - 0.5) / self.config.time_step)):
            pattern.update(self.config.time_step, 0.0, 0.0)
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

        pattern.update(self.config.time_step, 0.0, 0.0)

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
            pattern.update(self.config.time_step, 0.0, 0.0)

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


class HomingSineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()
        self.rng = np.random.default_rng(11)

    def test_bullets_converge_on_the_soul(self) -> None:
        pattern = HomingSinePattern(self.config)
        pattern.level = 1
        pattern.reset(self.rng)
        while not pattern.bullets():
            pattern.update(self.config.time_step, 96.0, 96.0)
        first = pattern.bullets()[0]
        start = hypot(first.x - 96.0, first.y - 96.0)
        for _ in range(int(1.0 / self.config.time_step)):
            pattern.update(self.config.time_step, 96.0, 96.0)

        self.assertTrue(any(bullet is first for bullet in pattern.bullets()))
        self.assertTrue(all(bullet.color == "white" for bullet in pattern.bullets()))
        self.assertLess(hypot(first.x - 96.0, first.y - 96.0), start)

    def test_heading_steers_toward_the_soul(self) -> None:
        pattern = HomingSinePattern(self.config)
        pattern.level = 2
        pattern.reset(self.rng)
        for _ in range(int(1.5 / self.config.time_step)):
            pattern.update(self.config.time_step, 96.0, 96.0)

        bullet = pattern.bullets()[0]
        to_soul = atan2(96.0 - bullet.y, 96.0 - bullet.x)
        heading = atan2(bullet.vy, bullet.vx)
        difference = abs((to_soul - heading + np.pi) % tau - np.pi)
        self.assertLess(difference, 1.0)

    def test_higher_level_spawns_faster_and_more_often(self) -> None:
        easy = HomingSinePattern(self.config)
        easy.level = 1
        hard = HomingSinePattern(self.config)
        hard.level = 4

        self.assertGreater(hard.speed, easy.speed)
        self.assertLess(hard.spawn_interval, easy.spawn_interval)


class DartVolleyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()
        self.rng = np.random.default_rng(12)

    def test_volley_fires_fast_darts_toward_the_soul(self) -> None:
        pattern = DartVolleyPattern(self.config)
        pattern.level = 1
        pattern.reset(self.rng)
        for _ in range(int(pattern.volley_interval / self.config.time_step) + 2):
            pattern.update(self.config.time_step, 96.0, 96.0)

        bullets = pattern.bullets()
        self.assertEqual(len(bullets), pattern.volley_size)
        self.assertTrue(all(bullet.color == "white" for bullet in bullets))
        self.assertTrue(all(bullet.vx > 100.0 for bullet in bullets))

    def test_higher_level_fires_bigger_faster_volleys(self) -> None:
        easy = DartVolleyPattern(self.config)
        easy.level = 1
        hard = DartVolleyPattern(self.config)
        hard.level = 4

        self.assertGreater(hard.volley_size, easy.volley_size)
        self.assertGreater(hard.speed, easy.speed)


class SpiralBarrageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()
        self.rng = np.random.default_rng(13)

    def test_bullets_fire_inward_from_the_edge(self) -> None:
        pattern = SpiralBarragePattern(self.config)
        pattern.level = 1
        pattern.reset(self.rng)
        for _ in range(10):
            pattern.update(self.config.time_step, 96.0, 96.0)

        center_x = self.config.board_width / 2
        center_y = self.config.board_height / 2
        self.assertGreater(len(pattern.bullets()), 0)
        for bullet in pattern.bullets():
            self.assertEqual(bullet.color, "white")
            inward = (center_x - bullet.x) * bullet.vx + (center_y - bullet.y) * bullet.vy
            self.assertGreater(inward, 0.0)

    def test_bullets_are_far_from_the_soul_at_spawn(self) -> None:
        pattern = SpiralBarragePattern(self.config)
        pattern.level = 1
        pattern.reset(self.rng)
        pattern.update(self.config.time_step, 96.0, 96.0)

        for bullet in pattern.bullets():
            self.assertGreater(hypot(bullet.x - 96.0, bullet.y - 96.0), 60.0)


class DrifterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()
        self.rng = np.random.default_rng(14)

    def test_drifters_stay_inside_the_board(self) -> None:
        pattern = DrifterPattern(self.config)
        pattern.level = 1
        pattern.reset(self.rng)

        for _ in range(int(20.0 / self.config.time_step)):
            pattern.update(self.config.time_step, 96.0, 96.0)

        self.assertEqual(len(pattern.bullets()), 8)
        for bullet in pattern.bullets():
            self.assertGreaterEqual(bullet.x, bullet.radius)
            self.assertLessEqual(bullet.x, self.config.board_width - bullet.radius)
            self.assertGreaterEqual(bullet.y, bullet.radius)
            self.assertLessEqual(bullet.y, self.config.board_height - bullet.radius)


class PickupsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()
        self.rng = np.random.default_rng(15)

    def test_pickups_spawn_green_away_from_the_soul(self) -> None:
        pattern = PickupsPattern(self.config)
        pattern.level = 1
        pattern.reset(self.rng)
        for _ in range(int(pattern.spawn_interval / self.config.time_step) + 2):
            pattern.update(self.config.time_step, 96.0, 96.0)

        bullets = pattern.bullets()
        self.assertGreater(len(bullets), 0)
        for bullet in bullets:
            self.assertEqual(bullet.color, "green")
            self.assertGreaterEqual(hypot(bullet.x - 96.0, bullet.y - 96.0), 45.0)


class BlueRainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()
        self.rng = np.random.default_rng(16)

    def test_blue_rain_falls_from_the_top(self) -> None:
        pattern = BlueRainPattern(self.config)
        pattern.level = 1
        pattern.reset(self.rng)
        for _ in range(40):
            pattern.update(self.config.time_step, 96.0, 96.0)

        bullets = pattern.bullets()
        self.assertGreater(len(bullets), 0)
        self.assertTrue(all(bullet.color == "blue" for bullet in bullets))
        self.assertTrue(all(bullet.vy > 0.0 for bullet in bullets))
        self.assertTrue(all(bullet.y > -self.config.soul_radius for bullet in bullets))


if __name__ == "__main__":
    unittest.main()
