import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball


class BlueRainPattern(BasePattern):
    """Utility: blue bullets raining from the top (only hurt a moving, unfocused SOUL)."""

    name = "blue_rain"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._spawn_timer = 0.0
        self._bullets: list[Fireball] = []

    @property
    def spawn_interval(self) -> float:
        return max(0.3, 1.0 / (1.0 + self.level))

    @property
    def fall_speed(self) -> float:
        return 45.0 + 8.0 * self.level

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._spawn_timer = 0.0
        self._bullets = []

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the blue rain.")
        self._spawn_timer += dt
        while self._spawn_timer >= self.spawn_interval:
            self._spawn_timer -= self.spawn_interval
            self._bullets.append(
                Fireball(
                    x=float(self._rng.uniform(self.config.soul_radius, self.config.board_width - self.config.soul_radius)),
                    y=-self.config.soul_radius,
                    vx=0.0,
                    vy=self.fall_speed,
                    radius=4.0,
                    color="blue",
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
