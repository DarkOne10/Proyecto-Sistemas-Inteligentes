import math

import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball

BURST_SIZE = 3
SHOT_SPACING = 0.12


class AimedBurstPattern(BasePattern):
    """A8: sniper bursts of three darts fired from a random edge, aimed at the SOUL."""

    name = "aimed_burst"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._burst_timer = 0.0
        self._pending: list[dict] = []
        self._bullets: list[Fireball] = []

    @property
    def burst_interval(self) -> float:
        return max(1.4, 2.6 - 0.15 * self.level)

    @property
    def speed(self) -> float:
        return 150.0 + 12.0 * self.level

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._burst_timer = 0.0
        self._pending = []
        self._bullets = []

    def _schedule_burst(self, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the aimed burst.")
        width = self.config.board_width
        height = self.config.board_height
        side = int(self._rng.integers(0, 4))
        if side == 0:
            x, y = float(self._rng.uniform(0.0, width)), -6.0
        elif side == 1:
            x, y = float(self._rng.uniform(0.0, width)), height + 6.0
        elif side == 2:
            x, y = -6.0, float(self._rng.uniform(0.0, height))
        else:
            x, y = width + 6.0, float(self._rng.uniform(0.0, height))
        aim = math.atan2(soul_y - y, soul_x - x)
        for index in range(BURST_SIZE):
            self._pending.append(
                {"delay": index * SHOT_SPACING, "x": x, "y": y, "angle": aim}
            )

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the aimed burst.")
        self._burst_timer += dt
        if self._burst_timer >= self.burst_interval:
            self._burst_timer = 0.0
            self._schedule_burst(soul_x, soul_y)

        still_pending = []
        for shot in self._pending:
            shot["delay"] -= dt
            if shot["delay"] <= 0.0:
                self._bullets.append(
                    Fireball(
                        x=shot["x"],
                        y=shot["y"],
                        vx=self.speed * math.cos(shot["angle"]),
                        vy=self.speed * math.sin(shot["angle"]),
                        radius=3.0,
                        color="white",
                    )
                )
            else:
                still_pending.append(shot)
        self._pending = still_pending

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
