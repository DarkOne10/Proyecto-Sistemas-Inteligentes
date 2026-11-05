import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball


class RainPattern(BasePattern):
    """Utility spawner: white bullets raining from the top, optionally ramping up."""

    name = "rain"
    damage = 4
    ramp_duration = 30.0

    def __init__(self, config, *, ramp: bool = False) -> None:
        super().__init__(config)
        self.ramp = ramp
        self.level = 1
        self._rng: np.random.Generator | None = None
        self._elapsed = 0.0
        self._spawn_acc = 0.0
        self._bullets: list[Fireball] = []

    @property
    def spawn_rate(self) -> float:
        if self.ramp:
            progress = min(self._elapsed / self.ramp_duration, 1.0)
            base_rate = 1.0 + 5.0 * progress
            return base_rate * (1.0 + 0.15 * (self.level - 1))
        return 1.0 + self.level

    @property
    def fall_speed(self) -> float:
        return 45.0 + 8.0 * self.level

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._elapsed = 0.0
        self._spawn_acc = 0.0
        self._bullets = []

    def update(self, dt: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the rain.")
        self._elapsed += dt
        self._spawn_acc += self.spawn_rate * dt
        while self._spawn_acc >= 1.0:
            self._spawn_acc -= 1.0
            self._bullets.append(
                Fireball(
                    x=float(self._rng.uniform(self.config.soul_radius, self.config.board_width - self.config.soul_radius)),
                    y=-self.config.soul_radius,
                    vx=0.0,
                    vy=self.fall_speed,
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
