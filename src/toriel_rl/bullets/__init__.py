from toriel_rl.bullets.base import BasePattern, EmptyPattern
from toriel_rl.bullets.blue_ring import BlueRingPattern
from toriel_rl.bullets.orange_gates import OrangeGatePattern
from toriel_rl.bullets.rain import RainPattern
from toriel_rl.bullets.rules import damages
from toriel_rl.bullets.sine_sweeper import SineSweeperPattern
from toriel_rl.config import EnvConfig

PATTERN_TYPES: dict[str, type[BasePattern]] = {
    "empty": EmptyPattern,
    "sine_sweeper": SineSweeperPattern,
    "blue_ring": BlueRingPattern,
    "orange_gates": OrangeGatePattern,
    "rain": RainPattern,
}


def create_pattern(name: str, config: EnvConfig) -> BasePattern:
    if name not in PATTERN_TYPES:
        raise ValueError(f"Unknown pattern: {name}")
    return PATTERN_TYPES[name](config)


__all__ = ["BasePattern", "EmptyPattern", "PATTERN_TYPES", "create_pattern", "damages"]
