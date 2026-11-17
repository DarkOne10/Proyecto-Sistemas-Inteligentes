import math

import numpy as np

from toriel_rl.attacks.base_attack import BaseAttack
from toriel_rl.entities.fireball import Fireball


class FalteringAttack(BaseAttack):
    name = "faltering"
    avoidance_radius = 30.0

    def __init__(self, config) -> None:
        super().__init__(config)
        self.fireballs: list[Fireball] = []
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self._reset_timer()
        self._rng = rng
        self.fireballs = [
            Fireball(
                x=float(rng.uniform(5, self.config.board_width - 5)),
                # Close enough above the board and fast enough to cross it within one round
                # (previously most fireballs were still mid-board when the round ended).
                y=float(rng.uniform(-0.25 * self.config.board_height, 0)),
                vx=float(rng.uniform(-20, 20)),
                vy=float(rng.uniform(56, 80)),
            )
            for _ in range(self.config.faltering_ball_count)
        ]

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        if self._rng is None:
            raise RuntimeError("Call reset() before updating an attack.")
        self._advance_timer(dt)

        for fireball in self.fireballs:
            dx = fireball.x - soul_x
            dy = fireball.y - soul_y
            distance = math.hypot(dx, dy)
            if 0 < distance < self.avoidance_radius:
                force = 100.0 * (1.0 - distance / self.avoidance_radius)
                fireball.vx += dx / distance * force * dt
                fireball.vy += dy / distance * force * dt
            elif distance == 0:
                angle = float(self._rng.uniform(0.0, math.tau))
                fireball.vx = math.cos(angle) * 40.0
                fireball.vy = math.sin(angle) * 40.0

            fireball.x += fireball.vx * dt
            fireball.y += fireball.vy * dt
            minimum_distance = fireball.radius + self.config.soul_radius + 1.0
            distance = math.hypot(fireball.x - soul_x, fireball.y - soul_y)
            if distance < minimum_distance:
                if distance == 0:
                    angle = float(self._rng.uniform(0.0, math.tau))
                    dx = math.cos(angle)
                    dy = math.sin(angle)
                else:
                    dx = (fireball.x - soul_x) / distance
                    dy = (fireball.y - soul_y) / distance
                fireball.x = soul_x + dx * minimum_distance
                fireball.y = soul_y + dy * minimum_distance
            if fireball.x < fireball.radius or fireball.x > self.config.board_width - fireball.radius:
                fireball.vx *= -1
                fireball.x = min(
                    max(fireball.x, fireball.radius),
                    self.config.board_width - fireball.radius,
                )
            if fireball.y > self.config.board_height + fireball.radius:
                fireball.x = float(self._rng.uniform(5, self.config.board_width - 5))
                fireball.y = -fireball.radius

    def collides(self, soul_x: float, soul_y: float, soul_radius: float) -> bool:
        return False