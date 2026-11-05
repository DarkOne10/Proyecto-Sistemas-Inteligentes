from toriel_rl.config import EnvConfig
from toriel_rl.scenarios.base import BaseScenario
from toriel_rl.scenarios.collapsing_arena import CollapsingArenaScenario
from toriel_rl.scenarios.narrow_corridor import NarrowCorridorScenario
from toriel_rl.scenarios.pinball import PinballScenario
from toriel_rl.scenarios.slalom_switching import SlalomSwitchingScenario
from toriel_rl.scenarios.spiral_evasion import SpiralEvasionScenario
from toriel_rl.scenarios.vertical_lanes import VerticalLanesScenario

SCENARIO_TYPES: dict[str, type[BaseScenario]] = {
    "vertical_lanes": VerticalLanesScenario,
    "spiral_evasion": SpiralEvasionScenario,
    "slalom_switching": SlalomSwitchingScenario,
    "pinball": PinballScenario,
    "narrow_corridor": NarrowCorridorScenario,
    "collapsing_arena": CollapsingArenaScenario,
}


def create_scenario(name: str, config: EnvConfig) -> BaseScenario:
    if name not in SCENARIO_TYPES:
        raise ValueError(f"Unknown scenario: {name}")
    return SCENARIO_TYPES[name](config)


__all__ = ["BaseScenario", "SCENARIO_TYPES", "create_scenario"]
