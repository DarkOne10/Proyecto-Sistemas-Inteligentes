import math

import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball

SPREAD_STEP = 0.12


class DartVolleyPattern(BasePattern):
    """A9: volleys of fast white darts aimed at the SOUL with a small angular spread."""

    name = "dart_volley"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._spawn_timer = 0.0
        self._bullets: list[Fireball] = []

    @property
    def volley_interval(self) -> float:
        return max(1.0, 2.2 - 0.2 * self.level)

    @property
    def volley_size(self) -> int:
        return 2 + self.level

    @property
    def speed(self) -> float:
        return 130.0 + 12.0 * self.level

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._spawn_timer = 0.0
        self._bullets = []

    def _volley(self, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the dart volley.")
        origin_y = float(self._rng.uniform(10.0, self.config.board_height - 10.0))
        aim = math.atan2(soul_y - origin_y, soul_x)
        count = self.volley_size
        for index in range(count):
            angle = aim + (index - (count - 1) / 2) * SPREAD_STEP
            self._bullets.append(
                Fireball(
                    x=-6.0,
                    y=origin_y,
                    vx=self.speed * math.cos(angle),
                    vy=self.speed * math.sin(angle),
                    radius=3.0,
                    color="white",
                )
            )

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the dart volley.")
        self._spawn_timer += dt
        if self._spawn_timer >= self.volley_interval:
            self._spawn_timer = 0.0
            self._volley(soul_x, soul_y)

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
