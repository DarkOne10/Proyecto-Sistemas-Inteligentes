import unittest

from gymnasium.utils.env_checker import check_env

from toriel_rl import EnvConfig, TorielEnv
from toriel_rl.entities.soul import Soul


class TorielEnvTests(unittest.TestCase):
    def test_environment_passes_gymnasium_checker(self) -> None:
        check_env(TorielEnv(), skip_render_check=True)

    def test_reset_is_reproducible_for_same_seed(self) -> None:
        env = TorielEnv()
        first_observation, first_info = env.reset(seed=17)
        second_observation, second_info = env.reset(seed=17)

        self.assertTrue((first_observation == second_observation).all())
        self.assertEqual(first_info["corridor_center"], second_info["corridor_center"])
        self.assertEqual(first_info["corridor_width"], second_info["corridor_width"])

    def test_different_seeds_change_the_attack_setup(self) -> None:
        env = TorielEnv()
        _, first_info = env.reset(seed=17)
        _, second_info = env.reset(seed=18)

        self.assertNotEqual(
            (first_info["corridor_center"], first_info["corridor_width"]),
            (second_info["corridor_center"], second_info["corridor_width"]),
        )

    def test_fire_helix_damage_switches_to_non_damaging_faltering(self) -> None:
        env = TorielEnv()
        _, info = env.reset(seed=9)
        env.soul.hp = 6
        env.soul.x = max(
            env.config.soul_radius,
            info["corridor_center"] - info["corridor_width"] / 2 - 1,
        )

        observation, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(info["damage"], 4)
        self.assertEqual(info["hp"], 2)
        self.assertEqual(info["attack"], "faltering")
        self.assertEqual(observation[3], 1.0)
        self.assertEqual(reward, -10.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        _, _, _, _, next_info = env.step(0)
        self.assertEqual(next_info["damage"], 0)
        self.assertEqual(next_info["hp"], 2)

    def test_low_hp_selects_faltering_without_another_hit(self) -> None:
        env = TorielEnv()
        env.reset(seed=5)
        env.soul.hp = 2

        observation, _, _, _, info = env.step(0)

        self.assertEqual(info["attack"], "faltering")
        self.assertEqual(observation[3], 1.0)

    def test_soul_movement_stays_inside_the_board(self) -> None:
        config = EnvConfig()
        soul = Soul(x=5, y=5, radius=config.soul_radius, hp=config.max_hp)

        soul.move(3, config.soul_speed, 1.0, config.board_width, config.board_height)
        soul.move(1, config.soul_speed, 1.0, config.board_width, config.board_height)

        self.assertEqual(soul.x, config.soul_radius)
        self.assertEqual(soul.y, config.soul_radius)

    def test_episode_limit_is_truncation_and_rewards_survival(self) -> None:
        config = EnvConfig(episode_seconds=1 / 30)
        env = TorielEnv(config=config)
        _, info = env.reset(seed=2)
        env.soul.x = info["corridor_center"]

        _, reward, terminated, truncated, info = env.step(0)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 5.0)
        self.assertEqual(info["reason"], "pattern_completed")


if __name__ == "__main__":
    unittest.main()