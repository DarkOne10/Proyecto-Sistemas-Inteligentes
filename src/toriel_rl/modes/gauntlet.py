import numpy as np

from toriel_rl.modes.base import BaseMode
from toriel_rl.modes.mixup import TRIVIAL_PATTERNS


class GauntletMode(BaseMode):
    """B6: generalist training — every episode is a random scene from the whole sandbox.

    With ``scenario_probability`` the episode is one of the arena scenarios (with its own
    default pattern); otherwise it is an open board with a random non-trivial pattern.
    The level is re-rolled too, so one policy has to cope with any attack it may meet.
    """

    name = "gauntlet"
    duration = 20.0
    scenario_probability = 0.35
    min_level = 1
    max_level = 3

    def reset(self, rng: np.random.Generator) -> None:
        self.level = int(rng.integers(self.min_level, self.max_level + 1))

    def select_scenario(self, rng: np.random.Generator, available: tuple[str, ...]) -> str | None:
        if not available or rng.random() >= self.scenario_probability:
            return None
        return str(rng.choice(available))

    def select_pattern(self, rng: np.random.Generator, available: tuple[str, ...]) -> str:
        pool = tuple(name for name in available if name not in TRIVIAL_PATTERNS)
        return str(rng.choice(pool))
