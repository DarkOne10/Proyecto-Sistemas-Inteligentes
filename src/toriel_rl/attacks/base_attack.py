from abc import ABC, abstractmethod

import numpy as np

from toriel_rl.config import EnvConfig
from toriel_rl.entities.fireball import Fireball


class BaseAttack(ABC):
    name: str
    damage: int = 0

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.elapsed = 0.0
        self.finished = False
        self.duration = config.episode_seconds

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def collides(self, soul_x: float, soul_y: float, soul_radius: float) -> bool:
        raise NotImplementedError

    def damage_at(self, soul_x: float, soul_y: float, soul_radius: float) -> int:
        return self.damage if self.collides(soul_x, soul_y, soul_radius) else 0

    def threats(self) -> list[Fireball]:
        return []

    def _reset_timer(self) -> None:
        self.elapsed = 0.0
        self.finished = False

    def _advance_timer(self, dt: float) -> None:
        self.elapsed += dt