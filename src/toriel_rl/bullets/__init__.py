from toriel_rl.bullets.base import BasePattern, EmptyPattern
from toriel_rl.bullets.rules import damages
from toriel_rl.config import EnvConfig

PATTERN_TYPES: dict[str, type[BasePattern]] = {"empty": EmptyPattern}


def create_pattern(name: str, config: EnvConfig) -> BasePattern:
    if name not in PATTERN_TYPES:
        raise ValueError(f"Unknown pattern: {name}")
    return PATTERN_TYPES[name](config)


__all__ = ["BasePattern", "EmptyPattern", "PATTERN_TYPES", "create_pattern", "damages"]
