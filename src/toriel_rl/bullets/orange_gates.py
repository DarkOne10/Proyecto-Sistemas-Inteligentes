import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball

WALL_SPACING = 12.0


class OrangeGatePattern(BasePattern):
    """A4: vertical orange walls with a gap, sweeping from left to right."""

    name = "orange_gates"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self.level = 1
        self._rng: np.random.Generator | None = None
        self._walls: list[dict] = []
        self._bullets: list[Fireball] = []

    @property
    def sweep_speed(self) -> float:
        return 40.0 + 5.0 * self.level

    @property
    def gap_half(self) -> float:
        return max(20.0, 28.0 - 2.0 * self.level)

    @property
    def wall_count(self) -> int:
        return 1 + self.level // 2

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._walls = []
        spacing = self.config.board_width / self.wall_count
        for index in range(self.wall_count):
            wall = {"x": -15.0 - index * spacing, "gap_y": 0.0}
            self._walls.append(wall)
            self._roll_gap(wall)
        self._rebuild_bullets()

    def _roll_gap(self, wall: dict) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating an orange gate.")
        margin = self.gap_half + 10.0
        wall["gap_y"] = float(self._rng.uniform(margin, self.config.board_height - margin))

    def _rebuild_bullets(self) -> None:
        self._bullets = []
        height = self.config.board_height
        for wall in self._walls:
            y = self.config.soul_radius
            while y <= height - self.config.soul_radius:
                if abs(y - wall["gap_y"]) > self.gap_half:
                    self._bullets.append(
                        Fireball(
                            x=wall["x"],
                            y=y,
                            vx=0.0,
                            vy=0.0,
                            radius=4.0,
                            color="orange",
                        )
                    )
                y += WALL_SPACING

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        width = self.config.board_width
        rebuilt = False
        for wall in self._walls:
            wall["x"] += self.sweep_speed * dt
            if wall["x"] > width + 6.0:
                wall["x"] = -6.0
                self._roll_gap(wall)
                rebuilt = True
        if rebuilt:
            self._rebuild_bullets()
        else:
            for bullet in self._bullets:
                bullet.x += self.sweep_speed * dt

    def bullets(self) -> list[Fireball]:
        return self._bullets
