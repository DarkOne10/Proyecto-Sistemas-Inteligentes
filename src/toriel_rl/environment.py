from math import ceil

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from toriel_rl.attacks.base_attack import BaseAttack
from toriel_rl.attacks.faltering import FalteringAttack
from toriel_rl.attacks.fire_helix import FireHelixAttack
from toriel_rl.config import EnvConfig
from toriel_rl.entities.soul import Soul


class TorielEnv(gym.Env):
    """A seeded, numeric-observation environment for dodging Toriel-inspired attacks."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )
        self.max_steps = ceil(self.config.episode_seconds / self.config.time_step)
        self.soul: Soul
        self.attack: BaseAttack
        self._scenario_center = self.config.board_width / 2
        self._scenario_width = self.config.corridor_width_min
        self._elapsed = 0.0
        self._steps = 0
        self._damage_cooldown = 0.0
        self._done = True
        self._surface = None
        self._pygame = None
        self._clock = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise ValueError("No reset options are supported yet.")

        self.soul = Soul(
            x=self.config.board_width / 2,
            y=self.config.board_height * 0.75,
            radius=self.config.soul_radius,
            hp=self.config.max_hp,
        )
        attack = FireHelixAttack(self.config)
        attack.reset(self.np_random)
        self.attack = attack
        self._scenario_center = attack.corridor_center
        self._scenario_width = attack.corridor_width
        self._elapsed = 0.0
        self._steps = 0
        self._damage_cooldown = 0.0
        self._done = False
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.soul.move(
            int(action),
            self.config.soul_speed,
            self.config.time_step,
            self.config.board_width,
            self.config.board_height,
        )
        self._steps += 1
        self._elapsed += self.config.time_step
        reward = 0.0
        damage = 0
        terminated = False
        truncated = False
        reason = None

        if isinstance(self.attack, FireHelixAttack) and self.soul.hp <= 2:
            self._activate_faltering()

        self.attack.update(self.config.time_step, self.soul.x, self.soul.y)
        self._damage_cooldown = max(0.0, self._damage_cooldown - self.config.time_step)

        if (
            isinstance(self.attack, FireHelixAttack)
            and self._damage_cooldown == 0.0
            and self.attack.collides(self.soul.x, self.soul.y, self.soul.radius)
        ):
            damage = min(self.config.fire_damage, self.soul.hp)
            self.soul.hp -= damage
            self._damage_cooldown = self.config.damage_cooldown
            reward -= 10.0
            if self.soul.hp == 0:
                reward -= 20.0
                terminated = True
                reason = "death"
            elif self.soul.hp <= 2:
                self._activate_faltering()

        if not terminated and self._steps >= self.max_steps:
            truncated = True
            reward += 5.0
            reason = "pattern_completed"

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
                (self.config.board_width * scale, self.config.board_height * scale)
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

    def _activate_faltering(self) -> None:
        attack = FalteringAttack(self.config)
        attack.reset(self.np_random)
        self.attack = attack

    def _get_observation(self) -> np.ndarray:
        attack_code = 1.0 if self.attack.name == "faltering" else 0.0
        phase = min(self._elapsed / self.config.episode_seconds, 1.0)
        return np.asarray(
            [
                self.soul.x / self.config.board_width,
                self.soul.y / self.config.board_height,
                self.soul.hp / self.config.max_hp,
                attack_code,
                phase,
                self._scenario_center / self.config.board_width,
                self._scenario_width / self.config.board_width,
            ],
            dtype=np.float32,
        )

    def _get_info(self, *, damage: int = 0, reason: str | None = None) -> dict:
        return {
            "seed": int(self.np_random_seed),
            "attack": self.attack.name,
            "hp": self.soul.hp,
            "damage": damage,
            "corridor_center": self._scenario_center,
            "corridor_width": self._scenario_width,
            "reason": reason,
        }