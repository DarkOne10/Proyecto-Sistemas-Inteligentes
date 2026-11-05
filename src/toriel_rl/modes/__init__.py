from toriel_rl.config import EnvConfig
from toriel_rl.modes.base import BaseMode, StepContext
from toriel_rl.modes.collector import CollectorMode
from toriel_rl.modes.focus_precision import FocusPrecisionMode
from toriel_rl.modes.survival import SurvivalMode

MODES: dict[str, type[BaseMode]] = {
    "survival": SurvivalMode,
    "collector": CollectorMode,
    "focus_precision": FocusPrecisionMode,
}


def create_mode(name: str, config: EnvConfig) -> BaseMode:
    if name not in MODES:
        raise ValueError(f"Unknown mode: {name}")
    return MODES[name](config)


def action_mode_for(name: str | None) -> str | None:
    if not name:
        return None
    if name not in MODES:
        raise ValueError(f"Unknown mode: {name}")
    return MODES[name].action_mode


__all__ = ["BaseMode", "MODES", "StepContext", "action_mode_for", "create_mode"]
