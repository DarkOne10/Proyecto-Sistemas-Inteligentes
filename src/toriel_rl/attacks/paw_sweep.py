import math

import numpy as np

from toriel_rl.attacks.base_attack import BaseAttack
from toriel_rl.entities.fireball import Fireball
from toriel_rl.entities.paw import Paw


class PawSweepAttack(BaseAttack):
    name = "paw_sweep"
    damage = 3
    paw_damage = 4

    def __init__(self, config, *, include_second_paw: bool = False) -> None:
        super().__init__(config)
        self.include_second_paw = include_second_paw
        self.paws: list[Paw] = []
        self.fireballs: list[Fireball] = []
        self._rng: np.random.Generator | None = None
        self._sweep_duration = min(config.paw_sweep_duration, config.episode_seconds * 0.55)
        self._next_trail_time = 0.0
        self._launched = False

    def reset(self, rng: np.random.Generator) -> None:
        self._reset_timer()
        self._rng = rng
        self._next_trail_time = 0.0
        self._launched = False
        self.fireballs = []
        self.paws = [Paw(0.12 * self.config.board_width, 0.18 * self.config.board_height)]
        if self.include_second_paw:
            self.paws.append(
                Paw(0.88 * self.config.board_width, 0.82 * self.config.board_height)
            )
        self._place_paws(0.0)

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        previous_elapsed = self.elapsed
        self._advance_timer(dt)
        progress = min(self.elapsed / self._sweep_duration, 1.0)
        self._place_paws(progress)

        while (
            previous_elapsed <= self._sweep_duration
            and self._next_trail_time <= min(self.elapsed, self._sweep_duration)
        ):
            trail_progress = min(self._next_trail_time / self._sweep_duration, 1.0)
            self._spawn_trail(trail_progress)
            self._next_trail_time += 0.16

        if not self._launched and self.elapsed >= self._sweep_duration:
            for fireball in self.fireballs:
                dx = soul_x - fireball.x
                dy = soul_y - fireball.y
                distance = max(1.0, math.hypot(dx, dy))
                speed = 72.0
                fireball.vx = dx / distance * speed
                fireball.vy = dy / distance * speed
            self._launched = True

        if self._launched:
            for fireball in self.fireballs:
                fireball.x += fireball.vx * dt
                fireball.y += fireball.vy * dt
                self._bounce(fireball)

    def collides(self, soul_x: float, soul_y: float, soul_radius: float) -> bool:
        return self.damage_at(soul_x, soul_y, soul_radius) > 0

    def damage_at(self, soul_x: float, soul_y: float, soul_radius: float) -> int:
        for paw in self.paws:
            if math.hypot(paw.x - soul_x, paw.y - soul_y) <= paw.radius + soul_radius:
                self.finished = True
                return self.paw_damage
        for fireball in self.fireballs:
            if math.hypot(fireball.x - soul_x, fireball.y - soul_y) <= fireball.radius + soul_radius:
                return self.damage
        return 0

    def threats(self) -> list[Fireball]:
        return self.fireballs

    def _place_paws(self, progress: float) -> None:
        width = self.config.board_width
        height = self.config.board_height
        if self.paws:
            self.paws[0].x = width * (0.12 + 0.76 * progress)
            self.paws[0].y = height * (0.18 + 0.35 * math.sin(math.pi * progress))
        if len(self.paws) > 1:
            self.paws[1].x = width * (0.88 - 0.76 * progress)
            self.paws[1].y = height * (0.82 - 0.35 * math.sin(math.pi * progress))

    def _spawn_trail(self, progress: float) -> None:
        width = self.config.board_width
        height = self.config.board_height
        if self.paws:
            self.fireballs.append(
                Fireball(
                    x=width * (0.12 + 0.76 * progress),
                    y=height * (0.18 + 0.35 * math.sin(math.pi * progress)),
                    vx=0.0,
                    vy=0.0,
                    radius=4.0,
                )
            )
        if len(self.paws) > 1:
            self.fireballs.append(
                Fireball(
                    x=width * (0.88 - 0.76 * progress),
                    y=height * (0.82 - 0.35 * math.sin(math.pi * progress)),
                    vx=0.0,
                    vy=0.0,
                    radius=4.0,
                )
            )

    def _bounce(self, fireball: Fireball) -> None:
        if fireball.x < fireball.radius or fireball.x > self.config.board_width - fireball.radius:
            fireball.vx *= -1
            fireball.x = min(
                max(fireball.x, fireball.radius),
                self.config.board_width - fireball.radius,
            )
        if fireball.y < fireball.radius or fireball.y > self.config.board_height - fireball.radius:
            fireball.vy *= -1
            fireball.y = min(
                max(fireball.y, fireball.radius),
                self.config.board_height - fireball.radius,
            )