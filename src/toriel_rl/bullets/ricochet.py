import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball


class RicochetPattern(BasePattern):
    """A7: fast angled white bullets that bounce off the side walls before leaving."""

    name = "ricochet"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._spawn_timer = 0.0
        self._bullets: list[Fireball] = []

    @property
    def spawn_interval(self) -> float:
        return max(0.5, 1.1 - 0.1 * self.level)

    @property
    def speed(self) -> float:
        return 110.0 + 10.0 * self.level

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._spawn_timer = 0.0
        self._bullets = []

    def _spawn(self) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the ricochet.")
        y = float(self._rng.uniform(20.0, self.config.board_height - 20.0))
        vy = float(self._rng.uniform(-35.0, 35.0))
        self._bullets.append(
            Fireball(
                x=-self.config.soul_radius,
                y=y,
                vx=self.speed,
                vy=vy,
                radius=4.0,
                color="white",
            )
        )

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the ricochet.")
        self._spawn_timer += dt
        while self._spawn_timer >= self.spawn_interval:
            self._spawn_timer -= self.spawn_interval
            self._spawn()

        width = self.config.board_width
        height = self.config.board_height
        keep = []
        for bullet in self._bullets:
            bullet.x += bullet.vx * dt
            bullet.y += bullet.vy * dt
            if bullet.y < bullet.radius or bullet.y > height - bullet.radius:
                bullet.vy = -bullet.vy
                bullet.y = min(max(bullet.y, bullet.radius), height - bullet.radius)
            if bullet.x >= width - bullet.radius and bullet.vx > 0.0:
                bullet.vx = -bullet.vx
                bullet.x = width - bullet.radius
            if -20.0 <= bullet.x:
                keep.append(bullet)
        self._bullets = keep

    def bullets(self) -> list[Fireball]:
        return self._bullets
