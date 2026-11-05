import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball
from toriel_rl.scenarios.base import BaseScenario

LANE_COUNT = 4


class VerticalLanesScenario(BaseScenario):
    """D1: rain restricted to lanes; one safe lane rotates on a timer."""

    name = "vertical_lanes"
    default_pattern = "empty"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._elapsed = 0.0
        self._spawn_timer = 0.0
        self._next_rotation = 0.0
        self._gap_lane = 0
        self._bullets: list[Fireball] = []

    @property
    def rotation_interval(self) -> float:
        return max(1.5, 4.0 - 0.5 * self.level)

    @property
    def spawn_interval(self) -> float:
        return max(0.18, 0.45 - 0.05 * self.level)

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._elapsed = 0.0
        self._spawn_timer = 0.0
        self._gap_lane = int(rng.integers(0, LANE_COUNT))
        self._next_rotation = self.rotation_interval
        self._bullets = []

    def _rotate_gap(self) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the lanes.")
        shift = int(self._rng.integers(1, LANE_COUNT))
        self._gap_lane = (self._gap_lane + shift) % LANE_COUNT

    def update(self, dt: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the lanes.")
        self._elapsed += dt
        if self._elapsed >= self._next_rotation:
            self._rotate_gap()
            self._next_rotation += self.rotation_interval

        self._spawn_timer += dt
        while self._spawn_timer >= self.spawn_interval:
            self._spawn_timer -= self.spawn_interval
            lane = int(self._rng.integers(0, LANE_COUNT - 1))
            if lane >= self._gap_lane:
                lane += 1
            lane_width = self.config.board_width / LANE_COUNT
            self._bullets.append(
                Fireball(
                    x=(lane + 0.5) * lane_width + float(self._rng.uniform(-8.0, 8.0)),
                    y=-self.config.soul_radius,
                    vx=0.0,
                    vy=50.0 + 8.0 * self.level,
                    radius=4.0,
                    color="white",
                )
            )

        remaining = []
        for bullet in self._bullets:
            bullet.y += bullet.vy * dt
            if bullet.y <= self.config.board_height + bullet.radius:
                remaining.append(bullet)
        self._bullets = remaining

    def bullets(self) -> list[Fireball]:
        return self._bullets
