from math import tau

import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball


class DrifterPattern(BasePattern):
    """Utility: slow white bullets drifting and bouncing off the walls (curriculum warm-up)."""

    name = "drifter"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._bullets: list[Fireball] = []

    @property
    def speed(self) -> float:
        return 22.0 + 4.0 * self.level

    def reset(self, rng: np.random.Generator) -> None:
        width = self.config.board_width
        height = self.config.board_height
        margin = 12.0
        self._bullets = []
        for _ in range(6 + 2 * self.level):
            angle = float(rng.uniform(0.0, tau))
            self._bullets.append(
                Fireball(
                    x=float(rng.uniform(margin, width - margin)),
                    y=float(rng.uniform(margin, height - margin)),
                    vx=self.speed * float(np.cos(angle)),
                    vy=self.speed * float(np.sin(angle)),
                    radius=4.0,
                    color="white",
                )
            )

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
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
