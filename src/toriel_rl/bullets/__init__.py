from toriel_rl.bullets.base import BasePattern, EmptyPattern
from toriel_rl.bullets.blue_rain import BlueRainPattern
from toriel_rl.bullets.blue_ring import BlueRingPattern
from toriel_rl.bullets.dart_volley import DartVolleyPattern
from toriel_rl.bullets.drifter import DrifterPattern
from toriel_rl.bullets.homing_sine import HomingSinePattern
from toriel_rl.bullets.orange_gates import OrangeGatePattern
from toriel_rl.bullets.pickups import PickupsPattern
from toriel_rl.bullets.rain import RainPattern
from toriel_rl.bullets.rules import damages
from toriel_rl.bullets.sine_sweeper import SineSweeperPattern
from toriel_rl.bullets.spiral_barrage import SpiralBarragePattern
from toriel_rl.config import EnvConfig

PATTERN_TYPES: dict[str, type[BasePattern]] = {
    "empty": EmptyPattern,
    "sine_sweeper": SineSweeperPattern,
    "blue_ring": BlueRingPattern,
    "orange_gates": OrangeGatePattern,
    "rain": RainPattern,
    "homing_sine": HomingSinePattern,
    "dart_volley": DartVolleyPattern,
    "spiral_barrage": SpiralBarragePattern,
    "drifter": DrifterPattern,
    "pickups": PickupsPattern,
    "blue_rain": BlueRainPattern,
}


def create_pattern(name: str, config: EnvConfig) -> BasePattern:
    if name not in PATTERN_TYPES:
        raise ValueError(f"Unknown pattern: {name}")
    return PATTERN_TYPES[name](config)


__all__ = ["BasePattern", "EmptyPattern", "PATTERN_TYPES", "create_pattern", "damages"]
