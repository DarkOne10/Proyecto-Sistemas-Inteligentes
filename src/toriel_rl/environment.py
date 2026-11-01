from math import ceil, hypot

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from toriel_rl.attacks.base_attack import BaseAttack
from toriel_rl.attacks.faltering import FalteringAttack
from toriel_rl.attacks.fire_helix import FireHelixAttack
from toriel_rl.attacks.fire_waves import FireWavesAttack
from toriel_rl.attacks.paw_sweep import PawSweepAttack
from toriel_rl.config import EnvConfig
from toriel_rl.entities.soul import Soul


class TorielEnv(gym.Env):
    """A seeded multi-round environment for dodging Toriel-inspired attacks."""

    metadata = {"render_modes": ["human"], "render_fps": 60}
    attack_codes = {"fire_helix": 0, "fire_waves": 1, "paw_sweep": 2, "faltering": 3}
    available_attacks = ("fire_helix", "fire_waves", "paw_sweep")

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
        self.steps_per_attack = ceil(self.config.episode_seconds / self.config.time_step)
        self.soul: Soul
        self.attack: BaseAttack
        self._scenario_center = self.config.board_width / 2
        self._scenario_width = self.config.board_width
        self._round_steps = 0
        self._round_index = 0
        self._spare_count = 0
        self._attack_only = False
        self._damage_cooldown = 0.0
        self._damage_this_round = False
        self._done = True
        self._surface = None
        self._pygame = None
        self._clock = None

    @property
    def spare_count(self) -> int:
        return self._spare_count

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        unsupported_options = set(options) - {"attack", "hp", "spares", "attack_only"}
        if unsupported_options:
            raise ValueError(f"Unsupported reset options: {sorted(unsupported_options)}")

        initial_hp = int(options.get("hp", self.config.max_hp))
        initial_spares = int(options.get("spares", 0))
        if not 1 <= initial_hp <= self.config.max_hp:
            raise ValueError("Initial HP must be between 1 and max_hp.")
        if not 0 <= initial_spares < self.config.max_spares:
            raise ValueError("Initial Spares must be below max_spares.")

        self.soul = Soul(
            x=self.config.board_width / 2,
            y=self.config.board_height * 0.75,
            radius=self.config.soul_radius,
            hp=initial_hp,
        )
        self._round_steps = 0
        self._round_index = initial_spares
        self._spare_count = initial_spares
        self._attack_only = bool(options.get("attack_only", False))
        self._damage_cooldown = 0.0
        self._damage_this_round = False
        self._done = False
        self._start_attack(options.get("attack"))
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        active_height = min(
            self.config.board_height,
            float(getattr(self.attack, "active_height", self.config.board_height)),
        )
        self.soul.move(
            int(action),
            self.config.soul_speed,
            self.config.time_step,
            self.config.board_width,
            active_height,
        )
        self._round_steps += 1
        self.attack.update(self.config.time_step, self.soul.x, self.soul.y)
        self._damage_cooldown = max(0.0, self._damage_cooldown - self.config.time_step)

        reward = 0.0
        damage = 0
        terminated = False
        truncated = False
        reason = None

        incoming_damage = self.attack.damage_at(self.soul.x, self.soul.y, self.soul.radius)
        if not incoming_damage:
            reward += 0.05
        if incoming_damage and (self._damage_cooldown == 0.0 or self.attack.finished):
            damage = min(incoming_damage, self.soul.hp)
            self.soul.hp -= damage
            self._damage_cooldown = self.config.damage_cooldown
            self._damage_this_round = True
            reward -= 10.0
            if self.soul.hp == 0:
                reward -= 20.0
                terminated = True
                reason = "death"
            elif self.soul.hp <= 2 and not isinstance(self.attack, FalteringAttack):
                self._start_attack("faltering")

        if not terminated and (self.attack.finished or self.attack.elapsed >= self.attack.duration):
            if self._attack_only:
                if not self._damage_this_round:
                    reward += 5.0
                truncated = True
                reason = "attack_completed"
            else:
                if not self._damage_this_round:
                    reward += 0.5
                self._spare_count += 1
                self._round_index += 1
                self._round_steps = 0
                self._damage_cooldown = 0.0
                self._damage_this_round = False
                if self._spare_count >= self.config.max_spares:
                    reward += 5.0
                    terminated = True
                    reason = "toriel_stopped"
                else:
                    self._start_attack()

        self._done = terminated or truncated
        info = self._get_info(damage=damage, reason=reason)
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        import pygame

        from toriel_rl.visualization import draw_environment

        if self._surface is None:
            pygame.init()
            scale = 3
            self._surface = pygame.display.set_mode(
                (self.config.board_width * scale, self.config.board_height * scale + 32)
            )
            pygame.display.set_caption("Toriel RL - entorno de esquiva")
            self._pygame = pygame
            self._clock = pygame.time.Clock()
        pygame.event.pump()
        draw_environment(self._surface, self, pygame)
        pygame.display.flip()
        self._clock.tick(max(1, round(1.0 / self.config.time_step)))
        return None

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.display.quit()
            self._pygame.quit()
        self._surface = None
        self._pygame = None
        self._clock = None

    def _start_attack(self, requested_name: str | None = None) -> None:
        forced_faltering = self.soul.hp <= 2 or (
            self._spare_count >= self.config.max_spares - 1 and self._spare_count > 0
        )
        name = "faltering" if forced_faltering else requested_name
        if name is None:
            name = str(self.np_random.choice(self.available_attacks))
        if name not in {*self.available_attacks, "faltering"}:
            raise ValueError(f"Unknown attack: {name}")
        if forced_faltering and name != "faltering":
            name = "faltering"

        if name == "fire_helix":
            attack: BaseAttack = FireHelixAttack(self.config)
        elif name == "fire_waves":
            attack = FireWavesAttack(self.config)
        elif name == "paw_sweep":
            include_second = self._round_index >= 2 and bool(self.np_random.integers(0, 2))
            attack = PawSweepAttack(self.config, include_second_paw=include_second)
        else:
            attack = FalteringAttack(self.config)

        attack.reset(self.np_random)
        self.attack = attack
        if isinstance(attack, FireHelixAttack):
            self._scenario_center = attack.corridor_center
            self._scenario_width = attack.corridor_width
        else:
            self._scenario_center = self.config.board_width / 2
            self._scenario_width = self.config.board_width

    def _get_observation(self) -> np.ndarray:
        attack_code = self.attack_codes[self.attack.name] / 3.0
        phase = min(self.attack.elapsed / max(self.attack.duration, 1e-6), 1.0)
        if isinstance(self.attack, FireHelixAttack):
            relative_x = (self.soul.x - self._scenario_center) / max(self._scenario_width, 1.0)
            corridor_position = float(np.clip(0.5 + relative_x * 0.25, 0.0, 1.0))
            corridor_width = self._scenario_width / self.config.board_width
        else:
            corridor_position = 0.5
            corridor_width = 1.0

        dangers = self._danger_distances()
        return np.asarray(
            [
                self.soul.x / self.config.board_width,
                self.soul.y / self.config.board_height,
                self.soul.hp / self.config.max_hp,
                attack_code,
                phase,
                self._spare_count / self.config.max_spares,
                corridor_position,
                corridor_width,
                *dangers,
            ],
            dtype=np.float32,
        )

    def _danger_distances(self) -> tuple[float, float, float, float]:
        diagonal = hypot(self.config.board_width, self.config.board_height)
        distances = [1.0, 1.0, 1.0, 1.0]
        if isinstance(self.attack, FireHelixAttack):
            left = max(0.0, self.soul.x - self.attack.corridor_left - self.soul.radius)
            right = max(0.0, self.attack.corridor_right - self.soul.x - self.soul.radius)
            if self.soul.x - self.soul.radius < self.attack.corridor_left:
                left = 0.0
            if self.soul.x + self.soul.radius > self.attack.corridor_right:
                right = 0.0
            distances[2] = min(distances[2], left / diagonal)
            distances[3] = min(distances[3], right / diagonal)

        sources = list(self.attack.threats())
        if isinstance(self.attack, PawSweepAttack):
            sources.extend(self.attack.paws)
        for source in sources:
            dx = source.x - self.soul.x
            dy = source.y - self.soul.y
            distance = max(0.0, hypot(dx, dy) - source.radius - self.soul.radius) / diagonal
            if abs(dx) > abs(dy):
                index = 2 if dx < 0 else 3
            else:
                index = 0 if dy < 0 else 1
            distances[index] = min(distances[index], distance)
        return tuple(float(np.clip(distance, 0.0, 1.0)) for distance in distances)

    def _get_info(self, *, damage: int = 0, reason: str | None = None) -> dict:
        return {
            "seed": int(self.np_random_seed),
            "attack": self.attack.name,
            "hp": self.soul.hp,
            "damage": damage,
            "spares": self._spare_count,
            "round": self._round_index + 1,
            "corridor_center": self._scenario_center,
            "corridor_width": self._scenario_width,
            "hard_mode": bool(getattr(self.attack, "hard_mode", False)),
            "second_paw": bool(getattr(self.attack, "include_second_paw", False)),
            "reason": reason,
        }