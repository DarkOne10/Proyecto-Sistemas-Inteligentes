import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from toriel_rl.bullets import EmptyPattern
from toriel_rl.bullets.base import BasePattern
from toriel_rl.bullets.rules import damages
from toriel_rl.config import EnvConfig
from toriel_rl.entities.fireball import Fireball
from toriel_rl.entities.soul import Soul
from toriel_rl.obs import BASE_FEATURE_COUNT, ObsConfig, observation_size
from toriel_rl.sandbox_env import SoulSandboxEnv


class StaticPattern(BasePattern):
    """Test helper: fixed bullet list with an optional external push."""

    name = "static"

    def __init__(self, config: EnvConfig, bullets: list[Fireball] | None = None, push=(0.0, 0.0)) -> None:
        super().__init__(config)
        self._bullets = list(bullets or [])
        self._push = push

    def reset(self, rng: np.random.Generator) -> None:
        return None

    def update(self, dt: float) -> None:
        return None

    def bullets(self) -> list[Fireball]:
        return self._bullets

    def external_displacement(self, dt: float) -> tuple[float, float]:
        return self._push


class ColorRuleTests(unittest.TestCase):
    def test_white_damages_regardless_of_movement(self) -> None:
        self.assertTrue(damages("white", moving=False, invulnerable=False))
        self.assertTrue(damages("white", moving=True, invulnerable=False))

    def test_blue_damages_only_while_moving(self) -> None:
        self.assertTrue(damages("blue", moving=True, invulnerable=False))
        self.assertFalse(damages("blue", moving=False, invulnerable=False))

    def test_orange_damages_only_while_stationary(self) -> None:
        self.assertTrue(damages("orange", moving=False, invulnerable=False))
        self.assertFalse(damages("orange", moving=True, invulnerable=False))

    def test_green_and_invulnerable_never_damage(self) -> None:
        self.assertFalse(damages("green", moving=True, invulnerable=False))
        self.assertFalse(damages("green", moving=False, invulnerable=False))
        self.assertFalse(damages("white", moving=True, invulnerable=True))
        self.assertFalse(damages("blue", moving=True, invulnerable=True))
        self.assertFalse(damages("orange", moving=False, invulnerable=True))


class SoulMovementTests(unittest.TestCase):
    def test_action_space_has_nine_directions(self) -> None:
        env = SoulSandboxEnv()

        self.assertEqual(env.action_space.n, 9)

    def test_diagonal_movement_is_speed_normalized(self) -> None:
        config = EnvConfig()
        soul = Soul(x=96, y=96, radius=config.soul_radius, hp=config.max_hp)

        soul.move(6, config.soul_speed, config.time_step, config.board_width, config.board_height)
        diagonal_step = np.hypot(soul.x - 96, soul.y - 96)

        soul2 = Soul(x=96, y=96, radius=config.soul_radius, hp=config.max_hp)
        soul2.move(4, config.soul_speed, config.time_step, config.board_width, config.board_height)
        cardinal_step = soul2.x - 96

        self.assertAlmostEqual(diagonal_step, cardinal_step, places=6)

    def test_legacy_cardinal_actions_keep_original_semantics(self) -> None:
        config = EnvConfig()
        soul = Soul(x=96, y=96, radius=config.soul_radius, hp=config.max_hp)

        soul.move(0, config.soul_speed, config.time_step, config.board_width, config.board_height)
        self.assertEqual((soul.x, soul.y), (96.0, 96.0))

        soul.move(4, config.soul_speed, config.time_step, config.board_width, config.board_height)
        self.assertEqual(soul.x, 96.0 + config.soul_speed * config.time_step)

        soul.move(1, config.soul_speed, config.time_step, config.board_width, config.board_height)
        self.assertEqual(soul.y, 96.0 - config.soul_speed * config.time_step)


class SoulSandboxEnvTests(unittest.TestCase):
    def test_environment_passes_gymnasium_checker(self) -> None:
        check_env(SoulSandboxEnv(), skip_render_check=True)

    def test_observation_shape_and_bounds(self) -> None:
        env = SoulSandboxEnv()
        observation, _ = env.reset(seed=1)

        self.assertEqual(env.observation_space.shape, (98,))
        self.assertTrue(env.observation_space.contains(observation))

    def test_mask_k_keeps_shape_and_zeroes_extra_slots(self) -> None:
        obs_config = ObsConfig(k_bullets=8, mask_k=3)
        env = SoulSandboxEnv(obs_config=obs_config)
        env.reset(seed=2, options={"pattern": "empty"})
        env.pattern = StaticPattern(
            env.config,
            bullets=[Fireball(x=20.0 + 15 * index, y=30.0, vx=0.0, vy=0.0) for index in range(8)],
        )

        observation, _, _, _, _ = env.step(0)
        slot_start = BASE_FEATURE_COUNT
        mask_start = BASE_FEATURE_COUNT + 3 * 10

        self.assertEqual(observation.shape, (98,))
        self.assertTrue(np.any(observation[slot_start:mask_start] != 0.0))
        self.assertTrue(np.all(observation[mask_start:] == 0.0))
        self.assertEqual(observation_size(obs_config), 98)

    def test_white_contact_triggers_iframes_and_blocks_next_hit(self) -> None:
        env = SoulSandboxEnv(config=EnvConfig(episode_seconds=10.0))
        env.reset(seed=3, options={"pattern": "empty"})
        env.pattern = StaticPattern(
            env.config, bullets=[Fireball(x=env.soul.x, y=env.soul.y, vx=0.0, vy=0.0)]
        )

        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(info["damage"], 4)
        self.assertEqual(info["hp"], 16)
        self.assertTrue(info["iframes"])
        self.assertAlmostEqual(reward, -9.98)
        self.assertFalse(terminated or truncated)

        _, _, _, _, info = env.step(0)
        self.assertEqual(info["damage"], 0)
        self.assertEqual(info["hp"], 16)

    def test_blue_bullet_respects_movement_state(self) -> None:
        env = SoulSandboxEnv(config=EnvConfig(episode_seconds=10.0))
        env.reset(seed=4, options={"pattern": "empty"})
        env.pattern = StaticPattern(
            env.config,
            bullets=[Fireball(x=env.soul.x, y=env.soul.y, vx=0.0, vy=0.0, color="blue")],
        )

        _, _, _, _, info = env.step(0)
        self.assertEqual(info["damage"], 0)
        self.assertEqual(info["hp"], 20)

        _, _, _, _, info = env.step(1)
        self.assertEqual(info["damage"], 4)
        self.assertEqual(info["hp"], 16)

    def test_orange_bullet_respects_stillness_state(self) -> None:
        env = SoulSandboxEnv(config=EnvConfig(episode_seconds=10.0))
        env.reset(seed=5, options={"pattern": "empty"})
        env.pattern = StaticPattern(
            env.config,
            bullets=[Fireball(x=env.soul.x, y=env.soul.y, vx=0.0, vy=0.0, color="orange")],
        )

        _, _, _, _, info = env.step(1)
        self.assertEqual(info["damage"], 0)
        self.assertEqual(info["hp"], 20)

        bullet = env.pattern.bullets()[0]
        bullet.x = env.soul.x
        bullet.y = env.soul.y
        _, _, _, _, info = env.step(0)
        self.assertEqual(info["damage"], 4)
        self.assertEqual(info["hp"], 16)

    def test_green_bullet_heals_and_is_consumed(self) -> None:
        env = SoulSandboxEnv(config=EnvConfig(episode_seconds=10.0))
        env.reset(seed=6, options={"pattern": "empty", "hp": 10})
        env.pattern = StaticPattern(
            env.config,
            bullets=[Fireball(x=env.soul.x, y=env.soul.y, vx=0.0, vy=0.0, color="green")],
        )

        _, reward, _, _, info = env.step(0)

        self.assertEqual(info["greens"], 1)
        self.assertEqual(info["hp"], 11)
        self.assertEqual(env.pattern.bullets(), [])
        self.assertAlmostEqual(reward, 1.02)

    def test_external_displacement_pushes_soul(self) -> None:
        env = SoulSandboxEnv(config=EnvConfig(episode_seconds=10.0))
        env.reset(seed=7, options={"pattern": "empty"})
        self.assertEqual(EmptyPattern(env.config).external_displacement(0.1), (0.0, 0.0))
        env.pattern = StaticPattern(env.config, push=(2.0, -1.0))

        env.step(0)

        self.assertAlmostEqual(env.soul.x, env.config.board_width / 2 + 2.0)
        self.assertAlmostEqual(env.soul.y, env.config.board_height / 2 - 1.0)

    def test_episode_truncates_with_clean_bonus(self) -> None:
        env = SoulSandboxEnv(config=EnvConfig(episode_seconds=1 / 30))
        env.reset(seed=8, options={"pattern": "empty"})

        _, reward, terminated, truncated, info = env.step(0)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reason"], "time_up")
        self.assertAlmostEqual(reward, 5.02)

    def test_lethal_hit_terminates_with_death_penalty(self) -> None:
        env = SoulSandboxEnv(config=EnvConfig(episode_seconds=10.0))
        env.reset(seed=9, options={"pattern": "empty", "hp": 1})
        env.pattern = StaticPattern(
            env.config, bullets=[Fireball(x=env.soul.x, y=env.soul.y, vx=0.0, vy=0.0)]
        )

        _, reward, terminated, truncated, info = env.step(0)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["hp"], 0)
        self.assertEqual(info["reason"], "death")
        self.assertAlmostEqual(reward, -30.0)

    def test_reset_is_reproducible(self) -> None:
        env = SoulSandboxEnv()
        first_observation, first_info = env.reset(seed=17)
        second_observation, second_info = env.reset(seed=17)

        self.assertTrue((first_observation == second_observation).all())
        self.assertEqual(first_info["pattern"], second_info["pattern"])

    def test_unknown_pattern_rejected(self) -> None:
        env = SoulSandboxEnv()

        with self.assertRaisesRegex(ValueError, "Unknown pattern"):
            env.reset(seed=1, options={"pattern": "laser"})

    def test_stepping_finished_episode_raises(self) -> None:
        env = SoulSandboxEnv(config=EnvConfig(episode_seconds=1 / 30))
        env.reset(seed=10, options={"pattern": "empty"})
        env.step(0)

        with self.assertRaises(RuntimeError):
            env.step(0)


if __name__ == "__main__":
    unittest.main()
