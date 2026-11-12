from math import ceil, hypot, sqrt

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from toriel_rl.bullets import BasePattern, PATTERN_TYPES, create_pattern
from toriel_rl.bullets.rules import damages
from toriel_rl.config import EnvConfig
from toriel_rl.entities.fireball import Fireball
from toriel_rl.entities.soul import Soul
from toriel_rl.modes import BaseMode, StepContext, action_mode_for, create_mode
from toriel_rl.obs import ObsConfig, build_observation, observation_size
from toriel_rl.rewards import RewardConfig
from toriel_rl.scenarios import BaseScenario, create_scenario

DODGE_GAP = 12.0
CONTINUOUS_DEADZONE = 0.1


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
        pattern: str | None = None,
        scenario: str | None = None,
        mode: str | None = None,
        action_mode: str | None = None,
        level: int = 1,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if level < 1:
            raise ValueError("Level must be positive.")

        self.config = config or EnvConfig()
        self.obs_config = obs_config or ObsConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.default_pattern = pattern
        self.default_scenario = scenario
        self.default_mode = mode
        self.default_level = level
        resolved_action_mode = action_mode or action_mode_for(mode) or "discrete"
        if resolved_action_mode not in ("discrete", "focus", "continuous"):
            raise ValueError(f"Unsupported action mode: {action_mode}")
        self.action_mode = resolved_action_mode
        if self.action_mode == "discrete":
            self.action_space = spaces.Discrete(9)
        elif self.action_mode == "focus":
            self.action_space = spaces.MultiDiscrete([9, 2])
        else:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_size(self.obs_config),),
            dtype=np.float32,
        )
        self.steps_per_episode = ceil(self.config.episode_seconds / self.config.time_step)
        self.soul: Soul
        self.pattern: BasePattern
        self.scenario: BaseScenario | None = None
        self.mode: BaseMode = BaseMode(self.config)
        self.level = level
        self.board_width = float(self.config.board_width)
        self.board_height = float(self.config.board_height)
        self.obstacles: list[tuple[float, float, float, float]] = []
        self._elapsed = 0.0
        self._steps = 0
        self._iframes = 0.0
        self._moving = False
        self._focus = False
        self._episode_damage = False
        self._greens_total = 0
        self._dodged = 0
        self._focus_steps = 0
        self._wall_hits = 0
        self._damage_by_color: dict[str, int] = {}
        self._tracked: dict[int, list] = {}
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
    def focus_active(self) -> bool:
        return self._focus

    @property
    def iframes_active(self) -> bool:
        return self._iframes > 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        unsupported_options = set(options) - {"pattern", "scenario", "mode", "level", "hp"}
        if unsupported_options:
            raise ValueError(f"Unsupported reset options: {sorted(unsupported_options)}")

        initial_hp = int(options.get("hp", self.config.max_hp))
        if not 1 <= initial_hp <= self.config.max_hp:
            raise ValueError("Initial HP must be between 1 and max_hp.")

        level = int(options.get("level", self.default_level))
        if level < 1:
            raise ValueError("Level must be positive.")

        mode_name = options.get("mode", self.default_mode)
        self.mode = create_mode(mode_name, self.config) if mode_name else BaseMode(self.config)
        if self.mode.action_mode and self.mode.action_mode != self.action_mode:
            raise ValueError(
                f"Mode '{self.mode.name}' requires action_mode={self.mode.action_mode!r}, "
                f"but this env uses {self.action_mode!r}."
            )
        self.mode.level = level
        self.mode.reset(self.np_random)
        level = self.mode.level

        scenario_name = options.get("scenario", self.default_scenario)
        self.scenario = create_scenario(scenario_name, self.config) if scenario_name else None
        if self.scenario is not None:
            self.scenario.level = level
            self.scenario.reset(self.np_random)

        pattern_name = options.get("pattern", self.default_pattern)
        if pattern_name is None and self.scenario is not None:
            pattern_name = self.scenario.default_pattern
        if pattern_name is None:
            pattern_name = self.mode.select_pattern(self.np_random, self.available_patterns)
        if pattern_name is None:
            pattern_name = self.mode.default_pattern
        if pattern_name is None:
            pattern_name = str(self.np_random.choice(self.available_patterns))

        self.pattern = create_pattern(pattern_name, self.config)
        self.pattern.level = level
        self.pattern.reset(self.np_random)

        self.soul = Soul(
            x=self.config.board_width / 2,
            y=self.config.board_height / 2,
            radius=self.config.soul_radius,
            hp=initial_hp,
        )
        self.level = level
        self.obstacles = list(self.mode.obstacles())
        if self.scenario is not None:
            self.obstacles.extend(self.scenario.obstacles())
        self._elapsed = 0.0
        self._steps = 0
        self._iframes = 0.0
        self._moving = False
        self._focus = False
        self._episode_damage = False
        self._greens_total = 0
        self._dodged = 0
        self._focus_steps = 0
        self._wall_hits = 0
        self._damage_by_color = {}
        self._tracked = {}
        self._done = False
        self.steps_per_episode = ceil(self.mode.duration_seconds / self.config.time_step)
        self._refresh_board()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        if self.action_mode == "focus":
            action = np.asarray(action, dtype=np.int64)
        elif self.action_mode == "continuous":
            action = np.asarray(action, dtype=np.float32).reshape(-1)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        focus = False
        move_action = 0
        dx = dy = 0.0
        if self.action_mode == "focus":
            move_action, focus = (int(part) for part in action)
            moving = move_action != 0
        elif self.action_mode == "continuous":
            dx, dy = float(action[0]), float(action[1])
            magnitude = hypot(dx, dy)
            if magnitude <= CONTINUOUS_DEADZONE:
                dx = dy = 0.0
                moving = False
            else:
                if magnitude > 1.0:
                    dx, dy = dx / magnitude, dy / magnitude
                moving = True
        else:
            move_action = int(action)
            moving = move_action != 0

        self._focus = focus
        if focus:
            self._focus_steps += 1
        speed = self.config.soul_speed * (0.5 if focus else 1.0)
        self._refresh_board()
        if self.action_mode == "continuous":
            self.soul.x = min(
                max(self.soul.x + dx * speed * self.config.time_step, self.soul.radius),
                self.board_width - self.soul.radius,
            )
            self.soul.y = min(
                max(self.soul.y + dy * speed * self.config.time_step, self.soul.radius),
                self.board_height - self.soul.radius,
            )
        else:
            self.soul.move(
                move_action, speed, self.config.time_step, self.board_width, self.board_height
            )
        self.pattern.update(self.config.time_step, self.soul.x, self.soul.y)
        if self.scenario is not None:
            self.scenario.update(self.config.time_step, self.soul.x, self.soul.y)

        push_x, push_y = self.pattern.external_displacement(
            self.config.time_step, self.soul.x, self.soul.y
        )
        if self.scenario is not None:
            scenario_x, scenario_y = self.scenario.external_displacement(
                self.config.time_step, self.soul.x, self.soul.y
            )
            push_x += scenario_x
            push_y += scenario_y
        if push_x or push_y:
            self.soul.x = min(max(self.soul.x + push_x, self.soul.radius), self.board_width - self.soul.radius)
            self.soul.y = min(max(self.soul.y + push_y, self.soul.radius), self.board_height - self.soul.radius)
        self._resolve_obstacles()

        self._elapsed += self.config.time_step
        self._steps += 1
        self._iframes = max(0.0, self._iframes - self.config.time_step)
        self._moving = moving
        damage, greens, nearest_gap = self._resolve_collisions(moving)
        self._greens_total += greens
        self._track_dodges()

        bullets = self.all_bullets()
        terminated = self.soul.hp <= 0
        reason = "death" if terminated else None
        truncated = False
        completion = None
        if not terminated:
            ctx = self._build_context(bullets, damage, greens, nearest_gap, moving, terminated)
            completion = self.mode.completion_reason(ctx)
            if completion:
                truncated = True
                reason = completion
            elif self._steps >= self.steps_per_episode:
                truncated = True
                reason = "time_up"

        reward = self._reward(damage=damage, greens=greens, terminated=terminated, truncated=truncated)
        if not terminated:
            reward += self.mode.extra_reward(ctx)
            if completion:
                reward += self.mode.completion_bonus(ctx)

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

    def _refresh_board(self) -> None:
        dynamic = self.mode.dynamic_level(self._elapsed)
        if dynamic is not None and dynamic > self.pattern.level:
            self.level = dynamic
            self.mode.level = dynamic
            self.pattern.level = dynamic
            if self.scenario is not None:
                self.scenario.level = dynamic
        size = None
        if self.mode is not None:
            size = self.mode.board_size(self._elapsed)
        if size is None and self.scenario is not None:
            size = self.scenario.board_size(self._elapsed)
        if size is None:
            size = (float(self.config.board_width), float(self.config.board_height))
        self.board_width, self.board_height = size
        self.soul.x = min(max(self.soul.x, self.soul.radius), self.board_width - self.soul.radius)
        self.soul.y = min(max(self.soul.y, self.soul.radius), self.board_height - self.soul.radius)

    def _resolve_obstacles(self) -> None:
        for ox, oy, ow, oh in self.obstacles:
            closest_x = min(max(self.soul.x, ox), ox + ow)
            closest_y = min(max(self.soul.y, oy), oy + oh)
            dx = self.soul.x - closest_x
            dy = self.soul.y - closest_y
            distance_sq = dx * dx + dy * dy
            radius = self.soul.radius
            if distance_sq >= radius * radius:
                continue
            self._wall_hits += 1
            if distance_sq > 0.0:
                distance = sqrt(distance_sq)
                push = (radius - distance) / distance
                self.soul.x += dx * push
                self.soul.y += dy * push
            else:
                left = self.soul.x - ox
                right = ox + ow - self.soul.x
                top = self.soul.y - oy
                bottom = oy + oh - self.soul.y
                nearest = min(left, right, top, bottom)
                if nearest == left:
                    self.soul.x = ox - radius
                elif nearest == right:
                    self.soul.x = ox + ow + radius
                elif nearest == top:
                    self.soul.y = oy - radius
                else:
                    self.soul.y = oy + oh + radius
            self.soul.x = min(max(self.soul.x, self.soul.radius), self.board_width - self.soul.radius)
            self.soul.y = min(max(self.soul.y, self.soul.radius), self.board_height - self.soul.radius)

    def all_bullets(self) -> list[Fireball]:
        bullets = list(self.pattern.bullets())
        if self.scenario is not None:
            bullets.extend(self.scenario.bullets())
        return bullets

    def _remove_bullet(self, bullet: Fireball) -> None:
        for owner in (self.pattern, self.scenario):
            if owner is None:
                continue
            try:
                owner.remove(bullet)
                return
            except ValueError:
                continue

    def _resolve_collisions(self, moving: bool) -> tuple[int, int, float]:
        damage = 0
        greens = 0
        nearest_gap = float("inf")
        invulnerable = self._iframes > 0.0
        damage_source = self.pattern
        if not self.pattern.bullets() and self.scenario is not None:
            damage_source = self.scenario
        for bullet in self.all_bullets():
            separation = hypot(bullet.x - self.soul.x, bullet.y - self.soul.y)
            gap = separation - bullet.radius - self.soul.radius
            if bullet.color != "green":
                nearest_gap = min(nearest_gap, gap)
            if gap > 0.0:
                continue
            if bullet.color == "green":
                self._remove_bullet(bullet)
                self.soul.hp = min(self.config.max_hp, self.soul.hp + self.config.green_heal)
                greens += 1
                continue
            if not damages(bullet.color, moving, invulnerable, self._focus):
                continue
            damage = min(damage_source.damage, self.soul.hp)
            self.soul.hp -= damage
            self._iframes = self.config.iframe_duration
            invulnerable = True
            self._episode_damage = True
            self._damage_by_color[bullet.color] = self._damage_by_color.get(bullet.color, 0) + damage
            tracked = self._tracked.get(id(bullet))
            if tracked is None:
                self._tracked[id(bullet)] = [bullet, True]
            else:
                tracked[1] = True
            if self.soul.hp == 0:
                break
        return damage, greens, nearest_gap

    def _track_dodges(self) -> None:
        bullets = self.all_bullets()
        for bullet in bullets:
            if bullet.color == "green":
                continue
            gap = hypot(bullet.x - self.soul.x, bullet.y - self.soul.y) - bullet.radius - self.soul.radius
            if gap < DODGE_GAP:
                self._tracked.setdefault(id(bullet), [bullet, False])
        alive = {id(bullet) for bullet in bullets}
        for key in [key for key in self._tracked if key not in alive]:
            _, hit = self._tracked.pop(key)
            if not hit:
                self._dodged += 1

    def _build_context(
        self,
        bullets: list[Fireball],
        damage: int,
        greens: int,
        nearest_gap: float,
        moving: bool,
        terminated: bool,
    ) -> StepContext:
        return StepContext(
            elapsed=self._elapsed,
            soul=self.soul,
            bullets=tuple(bullets),
            damage=damage,
            greens=greens,
            greens_total=self._greens_total,
            nearest_gap=nearest_gap,
            moving=moving,
            focus=self._focus,
            iframes=self._iframes > 0.0,
            whites_only=bool(bullets) and all(bullet.color == "white" for bullet in bullets),
            level=self.level,
            terminated=terminated,
        )

    def _reward(self, *, damage: int, greens: int, terminated: bool, truncated: bool) -> float:
        rewards = self.mode.reward_config() or self.reward_config
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
            self.all_bullets(),
            board_width=self.board_width,
            board_height=self.board_height,
            max_hp=self.config.max_hp,
            phase=self._elapsed / max(self.mode.duration_seconds, 1e-6),
            iframes=self._iframes > 0.0,
            moving=self._moving,
            focus=self._focus,
            config=self.obs_config,
        )

    def _get_info(self, *, damage: int = 0, greens: int = 0, reason: str | None = None) -> dict:
        return {
            "seed": int(self.np_random_seed),
            "pattern": self.pattern.name,
            "scenario": self.scenario.name if self.scenario is not None else None,
            "mode": self.mode.name,
            "level": self.level,
            "hp": self.soul.hp,
            "damage": damage,
            "greens": greens,
            "greens_total": self._greens_total,
            "bullets": len(self.all_bullets()),
            "iframes": self._iframes > 0.0,
            "moving": self._moving,
            "focus": self._focus,
            "action_mode": self.action_mode,
            "dodged": self._dodged,
            "focus_steps": self._focus_steps,
            "wall_hits": self._wall_hits,
            "damage_by_color": dict(self._damage_by_color),
            "objective": getattr(self.mode, "target", None),
            "reason": reason,
        }
