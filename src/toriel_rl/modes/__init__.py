from toriel_rl.config import EnvConfig
from toriel_rl.modes.base import BaseMode, StepContext
from toriel_rl.modes.survival import SurvivalMode

MODES: dict[str, type[BaseMode]] = {"survival": SurvivalMode}


def create_mode(name: str, config: EnvConfig) -> BaseMode:
    if name not in MODES:
        raise ValueError(f"Unknown mode: {name}")
    return MODES[name](config)


__all__ = ["BaseMode", "MODES", "StepContext", "create_mode"]
