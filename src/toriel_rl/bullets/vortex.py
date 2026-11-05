import math
from math import hypot

import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball


class VortexPattern(BasePattern):
    """A11: a rotational wind pushes the SOUL around the center while blue bullets orbit with it.
    The push is external force, so it never counts as movement: riding the vortex keeps blue safe."""

    name = "vortex"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._bullets: list[Fireball] = []

    @property
    def wind_speed(self) -> float:
        return 35.0 + 5.0 * self.level

    @property
    def bullet_speed(self) -> float:
        return 30.0 + 5.0 * self.level

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._bullets = []
        center_x = self.config.board_width / 2
        center_y = self.config.board_height / 2
        reach = min(self.config.board_width, self.config.board_height) * 0.45
        for _ in range(4 + 2 * self.level):
            angle = float(rng.uniform(0.0, math.tau))
            radius = float(rng.uniform(25.0, reach))
            self._bullets.append(
                Fireball(
                    x=center_x + radius * math.cos(angle),
                    y=center_y + radius * math.sin(angle),
                    vx=0.0,
                    vy=0.0,
                    radius=4.0,
                    color="blue",
                )
            )

    def external_displacement(self, dt: float, soul_x: float, soul_y: float) -> tuple[float, float]:
        center_x = self.config.board_width / 2
        center_y = self.config.board_height / 2
        dx = soul_x - center_x
        dy = soul_y - center_y
        distance = hypot(dx, dy)
        if distance < 1.0:
            return (0.0, 0.0)
        return (-dy / distance * self.wind_speed * dt, dx / distance * self.wind_speed * dt)

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        center_x = self.config.board_width / 2
        center_y = self.config.board_height / 2
        speed = self.bullet_speed
        for bullet in self._bullets:
            dx = bullet.x - center_x
            dy = bullet.y - center_y
            distance = hypot(dx, dy)
            if distance < 1.0:
                continue
            bullet.vx = -dy / distance * speed
            bullet.vy = dx / distance * speed
            bullet.x += bullet.vx * dt
            bullet.y += bullet.vy * dt

    def bullets(self) -> list[Fireball]:
        return self._bullets
