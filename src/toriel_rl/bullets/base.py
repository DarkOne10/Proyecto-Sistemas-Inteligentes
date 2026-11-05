from abc import ABC, abstractmethod

import numpy as np

from toriel_rl.config import EnvConfig
from toriel_rl.entities.fireball import Fireball


class BasePattern(ABC):
    """Self-contained bullet pattern for the Red SOUL sandbox."""

    name: str
    damage: int = 4
    level: int = 1

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def bullets(self) -> list[Fireball]:
        raise NotImplementedError

    def remove(self, bullet: Fireball) -> None:
        self.bullets().remove(bullet)

    def external_displacement(self, dt: float, soul_x: float, soul_y: float) -> tuple[float, float]:
        """Positional push applied to the SOUL after input movement (default: none)."""
        return (0.0, 0.0)


class EmptyPattern(BasePattern):
    """Neutral pattern without bullets, used as the sandbox default."""

    name = "empty"

    def reset(self, rng: np.random.Generator) -> None:
        return None

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        return None

    def bullets(self) -> list[Fireball]:
        return []
