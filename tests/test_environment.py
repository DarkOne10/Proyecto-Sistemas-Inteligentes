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

    def test_random_attack_selection_is_reproducible(self) -> None:
        env = TorielEnv()
        first_observation, first_info = env.reset(seed=73)
        second_observation, second_info = env.reset(seed=73)

        self.assertEqual(first_info["attack"], second_info["attack"])
        self.assertEqual(first_info["hard_mode"], second_info["hard_mode"])
        self.assertEqual(first_info["second_paw"], second_info["second_paw"])
        self.assertTrue((first_observation == second_observation).all())

    def test_different_seeds_change_the_attack_setup(self) -> None:
        env = TorielEnv()
        _, first_info = env.reset(seed=17, options={"attack": "fire_helix"})
        _, second_info = env.reset(seed=18, options={"attack": "fire_helix"})

        self.assertNotEqual(
            (first_info["corridor_center"], first_info["corridor_width"]),
            (second_info["corridor_center"], second_info["corridor_width"]),
        )

    def test_fire_helix_damage_switches_to_non_damaging_faltering(self) -> None:
        env = TorielEnv()
        _, info = env.reset(seed=9, options={"attack": "fire_helix"})
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
        env.reset(seed=5, options={"hp": 2, "attack": "fire_helix"})

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

    def test_attack_completion_advances_round_and_counts_spare(self) -> None:
        config = EnvConfig(episode_seconds=1 / 30)
        env = TorielEnv(config=config)
        _, info = env.reset(seed=2, options={"attack": "fire_helix"})
        env.soul.x = info["corridor_center"]

        _, reward, terminated, truncated, info = env.step(0)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 0.55)
        self.assertEqual(info["spares"], 1)
        self.assertEqual(info["round"], 2)
        self.assertIsNone(info["reason"])

    def test_all_documented_attacks_can_be_selected(self) -> None:
        env = TorielEnv()
        for attack_name in (*env.available_attacks, "faltering"):
            _, info = env.reset(seed=11, options={"attack": attack_name})
            self.assertEqual(info["attack"], attack_name)

    def test_paw_sweep_second_paw_appears_in_later_rounds(self) -> None:
        env = TorielEnv()
        second_paw_seen = False
        for seed in range(10):
            _, info = env.reset(seed=seed, options={"attack": "paw_sweep", "spares": 2})
            second_paw_seen = second_paw_seen or info["second_paw"]
        self.assertTrue(second_paw_seen)

    def test_eleven_spares_force_faltering_and_twelfth_ends_attack(self) -> None:
        config = EnvConfig(episode_seconds=1 / 30)
        env = TorielEnv(config=config)
        env.reset(seed=12, options={"spares": 10, "attack": "fire_helix"})
        env.soul.x = env.attack.corridor_center

        _, _, terminated, _, info = env.step(0)

        self.assertFalse(terminated)
        self.assertEqual(info["spares"], 11)
        self.assertEqual(info["attack"], "faltering")

        _, reward, terminated, truncated, info = env.step(0)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 5.55)
        self.assertEqual(info["spares"], 12)
        self.assertEqual(info["reason"], "toriel_stopped")

    def test_lethal_hit_terminates_with_death_reason(self) -> None:
        env = TorielEnv()
        _, info = env.reset(seed=9, options={"attack": "fire_helix", "hp": 4})
        env.soul.x = info["corridor_center"] - info["corridor_width"] / 2 - 1

        _, reward, terminated, truncated, info = env.step(0)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["hp"], 0)
        self.assertEqual(info["damage"], 4)
        self.assertEqual(info["reason"], "death")
        self.assertEqual(reward, -30.0)


if __name__ == "__main__":
    unittest.main()