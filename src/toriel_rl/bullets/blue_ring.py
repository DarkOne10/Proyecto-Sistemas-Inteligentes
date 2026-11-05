import math

import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball

RING_SIZE = 12


class BlueRingPattern(BasePattern):
    """A3: a ring of blue bullets contracting from the board edge toward the center."""

    name = "blue_ring"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self.level = 1
        self._elapsed = 0.0
        self._cycle = -1
        self._spin = 0.0
        self._bullets: list[Fireball] = []

    @property
    def contract_duration(self) -> float:
        return max(2.0, 5.0 - 0.5 * self.level)

    @property
    def start_radius(self) -> float:
        return min(self.config.board_width, self.config.board_height) * 0.45

    def reset(self, rng: np.random.Generator) -> None:
        self._elapsed = 0.0
        self._cycle = 0
        self._spin = float(rng.uniform(0.0, math.tau))
        self._bullets = self._make_ring()

    def _make_ring(self) -> list[Fireball]:
        center_x = self.config.board_width / 2
        center_y = self.config.board_height / 2
        return [
            Fireball(
                x=center_x,
                y=center_y,
                vx=0.0,
                vy=0.0,
                radius=4.0,
                color="blue",
                phase=index * math.tau / RING_SIZE,
            )
            for index in range(RING_SIZE)
        ]

    def update(self, dt: float) -> None:
        self._elapsed += dt
        cycle = int(self._elapsed / self.contract_duration)
        if cycle != self._cycle:
            self._cycle = cycle
            self._bullets = self._make_ring()
        progress = (self._elapsed % self.contract_duration) / self.contract_duration
        radius = self.start_radius * (1.0 - progress)
        center_x = self.config.board_width / 2
        center_y = self.config.board_height / 2
        spin = self._spin + 0.6 * self._elapsed
        for bullet in self._bullets:
            angle = bullet.phase + spin
            bullet.x = center_x + radius * math.cos(angle)
            bullet.y = center_y + radius * math.sin(angle)

    def bullets(self) -> list[Fireball]:
        return self._bullets
