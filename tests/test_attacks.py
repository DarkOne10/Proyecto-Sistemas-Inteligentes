import unittest

import numpy as np

from toriel_rl.attacks.faltering import FalteringAttack
from toriel_rl.attacks.fire_helix import FireHelixAttack
from toriel_rl.attacks.fire_waves import FireWavesAttack
from toriel_rl.attacks.paw_sweep import PawSweepAttack
from toriel_rl.config import EnvConfig
from toriel_rl.entities.fireball import Fireball


class AttackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvConfig()
        self.rng = np.random.default_rng(42)

    def test_fire_helix_has_a_safe_corridor_and_damaging_edges(self) -> None:
        attack = FireHelixAttack(self.config)
        attack.reset(self.rng)

        self.assertEqual(attack.damage_at(attack.corridor_center, 80, 5), 0)
        self.assertEqual(attack.damage_at(attack.corridor_left - 8, 80, 5), 4)

    def test_fire_waves_hard_mode_shortens_play_area_and_keeps_safe_corner(self) -> None:
        hard_config = EnvConfig(fire_waves_hard_probability=1.0)
        attack = FireWavesAttack(hard_config)
        attack.reset(self.rng)

        self.assertTrue(attack.hard_mode)
        self.assertLess(attack.active_height, hard_config.board_height)
        self.assertEqual(attack.damage_at(2, 2, hard_config.soul_radius), 0)

        fireball = attack.fireballs[0]
        self.assertEqual(
            attack.damage_at(fireball.x, fireball.y, hard_config.soul_radius),
            attack.damage,
        )

    def test_paw_contact_ends_attack_and_deals_four_damage(self) -> None:
        attack = PawSweepAttack(self.config)
        attack.reset(self.rng)
        paw = attack.paws[0]

        self.assertEqual(attack.damage_at(paw.x, paw.y, self.config.soul_radius), 4)
        self.assertTrue(attack.finished)

    def test_paw_fireball_deals_three_damage(self) -> None:
        attack = PawSweepAttack(self.config)
        attack.reset(self.rng)
        attack.paws.clear()
        attack.fireballs = [Fireball(x=40, y=40, vx=0, vy=0)]

        self.assertEqual(attack.damage_at(40, 40, self.config.soul_radius), 3)

    def test_faltering_never_deals_damage(self) -> None:
        attack = FalteringAttack(self.config)
        attack.reset(self.rng)
        attack.fireballs = [Fireball(x=40, y=40, vx=0, vy=0)]

        self.assertEqual(attack.damage_at(40, 40, self.config.soul_radius), 0)

    def test_faltering_moves_overlapping_fireball_outside_soul(self) -> None:
        attack = FalteringAttack(self.config)
        attack.reset(self.rng)
        attack.fireballs = [Fireball(x=40, y=40, vx=0, vy=0)]

        attack.update(self.config.time_step, soul_x=40, soul_y=40)

        fireball = attack.fireballs[0]
        distance = np.hypot(fireball.x - 40, fireball.y - 40)
        self.assertGreater(distance, fireball.radius + self.config.soul_radius)


if __name__ == "__main__":
    unittest.main()