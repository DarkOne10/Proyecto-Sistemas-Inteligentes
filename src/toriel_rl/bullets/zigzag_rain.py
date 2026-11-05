import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball


class ZigzagRainPattern(BasePattern):
    """A10: white bullets raining down while zigzagging between the side walls."""

    name = "zigzag_rain"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._spawn_timer = 0.0
        self._bullets: list[Fireball] = []

    @property
    def spawn_interval(self) -> float:
        return max(0.3, 0.6 - 0.04 * self.level)

    @property
    def fall_speed(self) -> float:
        return 50.0 + 6.0 * self.level

    @property
    def zig_speed(self) -> float:
        return 30.0 + 5.0 * self.level

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._spawn_timer = 0.0
        self._bullets = []

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the zigzag rain.")
        self._spawn_timer += dt
        while self._spawn_timer >= self.spawn_interval:
            self._spawn_timer -= self.spawn_interval
            self._bullets.append(
                Fireball(
                    x=float(self._rng.uniform(10.0, self.config.board_width - 10.0)),
                    y=-self.config.soul_radius,
                    vx=self.zig_speed * (1.0 if self._rng.random() < 0.5 else -1.0),
                    vy=self.fall_speed,
                    radius=4.0,
                    color="white",
                )
            )

        width = self.config.board_width
        remaining = []
        for bullet in self._bullets:
            bullet.x += bullet.vx * dt
            bullet.y += bullet.vy * dt
            if bullet.x < bullet.radius or bullet.x > width - bullet.radius:
                bullet.vx = -bullet.vx
                bullet.x = min(max(bullet.x, bullet.radius), width - bullet.radius)
            if bullet.y <= self.config.board_height + bullet.radius:
                remaining.append(bullet)
        self._bullets = remaining

    def bullets(self) -> list[Fireball]:
        return self._bullets
