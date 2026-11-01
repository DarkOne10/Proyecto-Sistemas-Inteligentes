from abc import ABC, abstractmethod

import numpy as np

from toriel_rl.config import EnvConfig


class BaseAttack(ABC):
    name: str

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def collides(self, soul_x: float, soul_y: float, soul_radius: float) -> bool:
        raise NotImplementedError