import math

import numpy as np

from toriel_rl.attacks.base_attack import BaseAttack
from toriel_rl.entities.fireball import Fireball


class FireWavesAttack(BaseAttack):
    name = "fire_waves"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self.fireballs: list[Fireball] = []
        self.hard_mode = False
        self.active_height = float(config.board_height)
        self.amplitude = config.board_width * 0.32

    @property
    def safe_width(self) -> float:
        return self.config.board_width * (0.11 if self.hard_mode else 0.19)

    @property
    def safe_height(self) -> float:
        return self.active_height * 0.24

    def reset(self, rng: np.random.Generator) -> None:
        self._reset_timer()
        self.hard_mode = bool(rng.random() < self.config.fire_waves_hard_probability)
        self.active_height = self.config.board_height * (0.78 if self.hard_mode else 1.0)
        self.amplitude = self.config.board_width * (0.38 if self.hard_mode else 0.32)
        self.fireballs = []
        row_count = 10 if self.hard_mode else 8
        spacing = self.active_height / row_count

        for lane in (-1, 1):
            for row in range(row_count + 2):
                if row % 5 == (0 if lane < 0 else 2):
                    continue
                self.fireballs.append(
                    Fireball(
                        x=self.config.board_width / 2,
                        y=(row - 1) * spacing,
                        vx=0.0,
                        vy=55.0 if self.hard_mode else 46.0,
                        radius=4.0 if self.hard_mode else 3.5,
                        lane=lane,
                        phase=float(rng.uniform(0.0, math.tau)),
                    )
                )

        if self.hard_mode:
            for x_ratio in (0.14, 0.86):
                for y_ratio in (0.35, 0.52, 0.69):
                    self.fireballs.append(
                        Fireball(
                            x=self.config.board_width * x_ratio,
                            y=self.active_height * y_ratio,
                            vx=0.0,
                            vy=0.0,
                            radius=5.0,
                        )
                    )
        self._update_stream_positions()

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        self._advance_timer(dt)
        for fireball in self.fireballs:
            if fireball.vy > 0:
                fireball.y += fireball.vy * dt
                if fireball.y > self.active_height + fireball.radius:
                    fireball.y = -fireball.radius
        self._update_stream_positions()

    def collides(self, soul_x: float, soul_y: float, soul_radius: float) -> bool:
        if self._inside_safe_corner(soul_x, soul_y, soul_radius):
            return False
        for fireball in self.fireballs:
            if math.hypot(fireball.x - soul_x, fireball.y - soul_y) <= fireball.radius + soul_radius:
                return True
        return False

    def threats(self) -> list[Fireball]:
        return self.fireballs

    def _inside_safe_corner(self, x: float, y: float, radius: float) -> bool:
        top_safe = y + radius <= self.safe_height
        return top_safe and (
            x + radius <= self.safe_width
            or x - radius >= self.config.board_width - self.safe_width
        )

    def _update_stream_positions(self) -> None:
        center = self.config.board_width / 2
        for fireball in self.fireballs:
            if fireball.vy == 0:
                continue
            vertical_phase = fireball.y / max(1.0, self.active_height) * math.tau
            angle = self.elapsed * 0.85 + vertical_phase + fireball.phase
            fireball.x = center + fireball.lane * self.amplitude * math.sin(angle)
            fireball.x = min(
                max(fireball.x, fireball.radius),
                self.config.board_width - fireball.radius,
            )