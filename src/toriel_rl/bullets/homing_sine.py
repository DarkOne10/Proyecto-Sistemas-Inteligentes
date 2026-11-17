import math

import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball


class HomingSinePattern(BasePattern):
    """A2: white bullets creeping in from the left, steering toward the SOUL with a sine weave."""

    name = "homing_sine"
    damage = 4
    # Bullets only steer for a while, then keep their heading and leave the board.
    # (Homing forever made them orbit the SOUL indefinitely: no policy could survive.)
    homing_seconds = 1.5

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._elapsed = 0.0
        self._spawn_timer = 0.0
        self._bullets: list[Fireball] = []
        self._angles: list[float] = []
        self._phases: list[float] = []
        self._ages: list[float] = []

    @property
    def spawn_interval(self) -> float:
        return max(0.5, 1.2 - 0.1 * self.level)

    @property
    def speed(self) -> float:
        return 50.0 + 6.0 * self.level

    @property
    def turn_rate(self) -> float:
        return 1.0 + 0.15 * self.level

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._elapsed = 0.0
        self._spawn_timer = 0.0
        self._bullets = []
        self._angles = []
        self._phases = []
        self._ages = []

    def _spawn(self, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the homing sine.")
        y = float(self._rng.uniform(self.config.soul_radius, self.config.board_height - self.config.soul_radius))
        aim = math.atan2(soul_y - y, soul_x)
        angle = aim + float(self._rng.uniform(-0.6, 0.6))
        self._bullets.append(
            Fireball(
                x=-self.config.soul_radius,
                y=y,
                vx=self.speed * math.cos(angle),
                vy=self.speed * math.sin(angle),
                radius=4.0,
                color="white",
            )
        )
        self._angles.append(angle)
        self._phases.append(float(self._rng.uniform(0.0, math.tau)))
        self._ages.append(0.0)

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the homing sine.")
        self._elapsed += dt
        self._spawn_timer += dt
        while self._spawn_timer >= self.spawn_interval:
            self._spawn_timer -= self.spawn_interval
            self._spawn(soul_x, soul_y)

        width = self.config.board_width
        height = self.config.board_height
        keep: list[int] = []
        for index, bullet in enumerate(self._bullets):
            self._ages[index] += dt
            if self._ages[index] <= self.homing_seconds:
                target = math.atan2(soul_y - bullet.y, soul_x - bullet.x)
                diff = (target - self._angles[index] + math.pi) % math.tau - math.pi
                turn = self.turn_rate * dt
                self._angles[index] += max(-turn, min(turn, diff))
                self._angles[index] += 1.6 * math.sin(2.5 * self._elapsed + self._phases[index]) * dt
            angle = self._angles[index]
            bullet.vx = self.speed * math.cos(angle)
            bullet.vy = self.speed * math.sin(angle)
            bullet.x += bullet.vx * dt
            bullet.y += bullet.vy * dt
            if -20.0 <= bullet.x <= width + 20.0 and -20.0 <= bullet.y <= height + 20.0:
                keep.append(index)
        self._bullets = [self._bullets[index] for index in keep]
        self._angles = [self._angles[index] for index in keep]
        self._phases = [self._phases[index] for index in keep]
        self._ages = [self._ages[index] for index in keep]

    def bullets(self) -> list[Fireball]:
        return self._bullets
