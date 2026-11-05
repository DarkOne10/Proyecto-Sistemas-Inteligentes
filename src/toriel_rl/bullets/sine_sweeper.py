import math

import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball


class SineSweeperPattern(BasePattern):
    """A1: a row of white bullets crossing the board with a vertical sine wobble."""

    name = "sine_sweeper"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self.level = 1
        self._elapsed = 0.0
        self._base_y: list[float] = []
        self._phase: list[float] = []
        self._bullets: list[Fireball] = []

    def reset(self, rng: np.random.Generator) -> None:
        self._elapsed = 0.0
        count = 4 + self.level
        height = self.config.board_height
        speed = 55.0 + 8.0 * self.level
        amplitude = height * 0.25
        self._bullets = []
        self._base_y = []
        self._phase = []
        for index in range(count):
            base_y = height * (index + 0.5) / count
            phase = float(rng.uniform(0.0, math.tau))
            y = min(max(base_y + amplitude * math.sin(phase), 4.0), height - 4.0)
            self._bullets.append(
                Fireball(
                    x=-float(rng.uniform(0.0, 40.0)),
                    y=y,
                    vx=speed,
                    vy=0.0,
                    radius=4.0,
                    color="white",
                )
            )
            self._base_y.append(base_y)
            self._phase.append(phase)

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        self._elapsed += dt
        amplitude = self.config.board_height * 0.25
        angular_speed = math.tau / 3.0
        width = self.config.board_width
        for index, bullet in enumerate(self._bullets):
            bullet.x += bullet.vx * dt
            if bullet.x > width + bullet.radius:
                bullet.x = -bullet.radius
            target = self._base_y[index] + amplitude * math.sin(
                angular_speed * self._elapsed + self._phase[index]
            )
            bullet.y = min(max(target, bullet.radius), self.config.board_height - bullet.radius)

    def bullets(self) -> list[Fireball]:
        return self._bullets
