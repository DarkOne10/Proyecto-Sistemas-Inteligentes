from math import tau

import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball

MIN_SPAWN_DISTANCE = 45.0


class PickupsPattern(BasePattern):
    """Utility: drifting green pickups spawned away from the SOUL (collector mode fodder)."""

    name = "pickups"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._spawn_timer = 0.0
        self._bullets: list[Fireball] = []

    @property
    def spawn_interval(self) -> float:
        return max(1.2, 2.5 - 0.2 * self.level)

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._spawn_timer = 0.0
        self._bullets = []

    def _spawn(self, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the pickups.")
        width = self.config.board_width
        height = self.config.board_height
        x = y = 0.0
        for _ in range(10):
            x = float(self._rng.uniform(12.0, width - 12.0))
            y = float(self._rng.uniform(12.0, height - 12.0))
            if (x - soul_x) ** 2 + (y - soul_y) ** 2 >= MIN_SPAWN_DISTANCE**2:
                break
        angle = float(self._rng.uniform(0.0, tau))
        drift = 10.0
        self._bullets.append(
            Fireball(
                x=x,
                y=y,
                vx=drift * float(np.cos(angle)),
                vy=drift * float(np.sin(angle)),
                radius=5.0,
                color="green",
            )
        )

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the pickups.")
        self._spawn_timer += dt
        if self._spawn_timer >= self.spawn_interval:
            self._spawn_timer = 0.0
            self._spawn(soul_x, soul_y)

        for bullet in self._bullets:
            bullet.x += bullet.vx * dt
            bullet.y += bullet.vy * dt
            if bullet.x < bullet.radius or bullet.x > self.config.board_width - bullet.radius:
                bullet.vx = -bullet.vx
                bullet.x = min(max(bullet.x, bullet.radius), self.config.board_width - bullet.radius)
            if bullet.y < bullet.radius or bullet.y > self.config.board_height - bullet.radius:
                bullet.vy = -bullet.vy
                bullet.y = min(max(bullet.y, bullet.radius), self.config.board_height - bullet.radius)

    def bullets(self) -> list[Fireball]:
        return self._bullets
