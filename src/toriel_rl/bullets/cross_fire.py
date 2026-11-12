import numpy as np

from toriel_rl.bullets.base import BasePattern
from toriel_rl.entities.fireball import Fireball


class CrossFirePattern(BasePattern):
    """A6: white bullets crossing the board from opposite sides in interleaved rows."""

    name = "cross_fire"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._spawn_timer = 0.0
        self._side = 0
        self._bullets: list[Fireball] = []

    @property
    def spawn_interval(self) -> float:
        return max(0.35, 0.7 - 0.05 * self.level)

    @property
    def speed(self) -> float:
        return 60.0 + 7.0 * self.level

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._spawn_timer = 0.0
        self._side = 0
        self._bullets = []

    def _spawn(self) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the cross fire.")
        y = float(self._rng.uniform(self.config.soul_radius, self.config.board_height - self.config.soul_radius))
        if self._side == 0:
            x, vx = -self.config.soul_radius, self.speed
        else:
            x, vx = self.config.board_width + self.config.soul_radius, -self.speed
        self._side = 1 - self._side
        self._bullets.append(
            Fireball(x=x, y=y, vx=vx, vy=0.0, radius=4.0, color="white")
        )

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating the cross fire.")
        self._spawn_timer += dt
        while self._spawn_timer >= self.spawn_interval:
            self._spawn_timer -= self.spawn_interval
            self._spawn()

        width = self.config.board_width
        for bullet in self._bullets:
            bullet.x += bullet.vx * dt
        self._bullets = [bullet for bullet in self._bullets if -20.0 <= bullet.x <= width + 20.0]

    def bullets(self) -> list[Fireball]:
        return self._bullets
