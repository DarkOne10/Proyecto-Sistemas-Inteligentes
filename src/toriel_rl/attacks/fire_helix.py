import numpy as np

from toriel_rl.attacks.base_attack import BaseAttack
from toriel_rl.entities.fireball import Fireball


class FireHelixAttack(BaseAttack):
    name = "fire_helix"
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self.corridor_center = config.board_width / 2
        self.corridor_width = config.corridor_width_min

    @property
    def corridor_left(self) -> float:
        return self.corridor_center - self.corridor_width / 2

    @property
    def corridor_right(self) -> float:
        return self.corridor_center + self.corridor_width / 2

    def reset(self, rng: np.random.Generator) -> None:
        self._reset_timer()
        self.corridor_width = float(
            rng.uniform(self.config.corridor_width_min, self.config.corridor_width_max)
        )
        min_center = self.config.soul_radius + self.corridor_width / 2
        max_center = self.config.board_width - min_center
        self.corridor_center = float(rng.uniform(min_center, max_center))

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        self._advance_timer(dt)

    def collides(self, soul_x: float, soul_y: float, soul_radius: float) -> bool:
        return (
            soul_x - soul_radius < self.corridor_left
            or soul_x + soul_radius > self.corridor_right
        )

    def threats(self) -> list[Fireball]:
        return []