import numpy as np

from toriel_rl.modes.base import BaseMode

TRIVIAL_PATTERNS = ("empty", "pickups")


class MixupMode(BaseMode):
    """B5: domain randomization — every episode rolls a fresh pattern and level."""

    name = "mixup"
    duration = 30.0

    def reset(self, rng: np.random.Generator) -> None:
        self.level = int(rng.integers(1, 4))

    def select_pattern(self, rng: np.random.Generator, available: tuple[str, ...]) -> str:
        pool = tuple(name for name in available if name not in TRIVIAL_PATTERNS)
        return str(rng.choice(pool))
