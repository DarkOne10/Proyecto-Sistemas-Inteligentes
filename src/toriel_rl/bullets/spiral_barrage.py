import math

import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball


class SpiralBarragePattern(BasePattern):
    """A5: a rotating lighthouse on the board edge firing white bullets inward across the arena."""

    name = "spiral_barrage"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._elapsed = 0.0
        self._phase = 0.0
        self._spawn_timer = 0.0
        self._bullets: list[Fireball] = []

    @property
    def spin_speed(self) -> float:
        return 0.9 + 0.25 * self.level

    @property
    def spawn_interval(self) -> float:
        return max(0.05, 0.12 - 0.01 * self.level)

    @property
    def speed(self) -> float:
        return 65.0 + 8.0 * self.level

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._elapsed = 0.0
        self._phase = float(rng.uniform(0.0, math.tau))
        self._spawn_timer = 0.0
        self._bullets = []

    def _edge_point(self, angle: float) -> tuple[float, float]:
        width = self.config.board_width
        height = self.config.board_height
        dx = math.cos(angle)
        dy = math.sin(angle)
        t_x = (width / 2) / abs(dx) if dx else math.inf
        t_y = (height / 2) / abs(dy) if dy else math.inf
        t = min(t_x, t_y) - 6.0
        return width / 2 + dx * t, height / 2 + dy * t

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        self._elapsed += dt
        self._spawn_timer += dt
        while self._spawn_timer >= self.spawn_interval:
            self._spawn_timer -= self.spawn_interval
            angle = self._phase + self.spin_speed * self._elapsed
            x, y = self._edge_point(angle)
            self._bullets.append(
                Fireball(
                    x=x,
                    y=y,
                    vx=-self.speed * math.cos(angle),
                    vy=-self.speed * math.sin(angle),
                    radius=4.0,
                    color="white",
                )
            )

        width = self.config.board_width
        height = self.config.board_height
        for bullet in self._bullets:
            bullet.x += bullet.vx * dt
            bullet.y += bullet.vy * dt
        self._bullets = [
            bullet
            for bullet in self._bullets
            if -20.0 <= bullet.x <= width + 20.0 and -20.0 <= bullet.y <= height + 20.0
        ]

    def bullets(self) -> list[Fireball]:
        return self._bullets
