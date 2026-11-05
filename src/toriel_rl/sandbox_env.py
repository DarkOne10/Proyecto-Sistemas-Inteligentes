from math import ceil, hypot

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from toriel_rl.bullets import BasePattern, PATTERN_TYPES, create_pattern
from toriel_rl.bullets.rules import damages
from toriel_rl.config import EnvConfig
from toriel_rl.entities.soul import Soul
from toriel_rl.obs import ObsConfig, build_observation, observation_size
from toriel_rl.rewards import RewardConfig


class SoulSandboxEnv(gym.Env):
    """Red SOUL sandbox: dodge color-coded bullets inside a neutral Bullet Board."""

    metadata = {"render_modes": ["human"], "render_fps": 60}
    available_patterns = tuple(PATTERN_TYPES)

    def __init__(
        self,
        config: EnvConfig | None = None,
        obs_config: ObsConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.config = config or EnvConfig()
        self.obs_config = obs_config or ObsConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_size(self.obs_config),),
            dtype=np.float32,
        )
        self.steps_per_episode = ceil(self.config.episode_seconds / self.config.time_step)
        self.soul: Soul
        self.pattern: BasePattern
        self._elapsed = 0.0
        self._steps = 0
        self._iframes = 0.0
        self._moving = False
        self._episode_damage = False
        self._done = True
        self._surface = None
        self._pygame = None
        self._clock = None

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def moving(self) -> bool:
        return self._moving

    @property
    def iframes_active(self) -> bool:
        return self._iframes > 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        unsupported_options = set(options) - {"pattern", "hp"}
        if unsupported_options:
            raise ValueError(f"Unsupported reset options: {sorted(unsupported_options)}")

        initial_hp = int(options.get("hp", self.config.max_hp))
        if not 1 <= initial_hp <= self.config.max_hp:
            raise ValueError("Initial HP must be between 1 and max_hp.")

        pattern_name = options.get("pattern")
        if pattern_name is None:
            pattern_name = str(self.np_random.choice(self.available_patterns))

        self.pattern = create_pattern(pattern_name, self.config)
        self.pattern.reset(self.np_random)
        self.soul = Soul(
            x=self.config.board_width / 2,
            y=self.config.board_height / 2,
            radius=self.config.soul_radius,
            hp=initial_hp,
        )
        self._elapsed = 0.0
        self._steps = 0
        self._iframes = 0.0
        self._moving = False
        self._episode_damage = False
        self._done = False
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        moving = action != 0
        width = self.config.board_width
        height = self.config.board_height
        self.soul.move(
            int(action), self.config.soul_speed, self.config.time_step, width, height
        )
        self.pattern.update(self.config.time_step)
        push_x, push_y = self.pattern.external_displacement(self.config.time_step)
        if push_x or push_y:
            self.soul.x = min(max(self.soul.x + push_x, self.soul.radius), width - self.soul.radius)
            self.soul.y = min(max(self.soul.y + push_y, self.soul.radius), height - self.soul.radius)

        self._elapsed += self.config.time_step
        self._steps += 1
        self._iframes = max(0.0, self._iframes - self.config.time_step)
        self._moving = moving
        damage, greens = self._resolve_collisions(moving)

        terminated = self.soul.hp <= 0
        truncated = False
        reason = "death" if terminated else None
        if not terminated and self._steps >= self.steps_per_episode:
            truncated = True
            reason = "time_up"

        reward = self._reward(damage=damage, greens=greens, terminated=terminated, truncated=truncated)
        self._done = terminated or truncated
        info = self._get_info(damage=damage, greens=greens, reason=reason)
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        import pygame

        from toriel_rl.visualization import draw_sandbox

        if self._surface is None:
            pygame.init()
            scale = 3
            self._surface = pygame.display.set_mode(
                (self.config.board_width * scale, self.config.board_height * scale + 32)
            )
            pygame.display.set_caption("Toriel RL - Red SOUL sandbox")
            self._pygame = pygame
            self._clock = pygame.time.Clock()
        pygame.event.pump()
        draw_sandbox(self._surface, self, pygame)
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

    def _resolve_collisions(self, moving: bool) -> tuple[int, int]:
        damage = 0
        greens = 0
        invulnerable = self._iframes > 0.0
        for bullet in list(self.pattern.bullets()):
            separation = hypot(bullet.x - self.soul.x, bullet.y - self.soul.y)
            if separation > bullet.radius + self.soul.radius:
                continue
            if bullet.color == "green":
                self.pattern.remove(bullet)
                self.soul.hp = min(self.config.max_hp, self.soul.hp + self.config.green_heal)
                greens += 1
                continue
            if not damages(bullet.color, moving, invulnerable):
                continue
            damage = min(self.pattern.damage, self.soul.hp)
            self.soul.hp -= damage
            self._iframes = self.config.iframe_duration
            invulnerable = True
            self._episode_damage = True
            if self.soul.hp == 0:
                break
        return damage, greens

    def _reward(self, *, damage: int, greens: int, terminated: bool, truncated: bool) -> float:
        rewards = self.reward_config
        reward = 0.0
        if greens:
            reward += rewards.green * greens
        if damage:
            reward += rewards.damage
        if terminated:
            reward += rewards.death
        else:
            reward += rewards.survival
            if truncated and not self._episode_damage:
                reward += rewards.clean_completion
        return reward

    def _get_observation(self) -> np.ndarray:
        return build_observation(
            self.soul,
            self.pattern.bullets(),
            board_width=self.config.board_width,
            board_height=self.config.board_height,
            max_hp=self.config.max_hp,
            phase=self._elapsed / max(self.config.episode_seconds, 1e-6),
            iframes=self._iframes > 0.0,
            moving=self._moving,
            focus=False,
            config=self.obs_config,
        )

    def _get_info(self, *, damage: int = 0, greens: int = 0, reason: str | None = None) -> dict:
        return {
            "seed": int(self.np_random_seed),
            "pattern": self.pattern.name,
            "hp": self.soul.hp,
            "damage": damage,
            "greens": greens,
            "bullets": len(self.pattern.bullets()),
            "iframes": self._iframes > 0.0,
            "moving": self._moving,
            "reason": reason,
        }
