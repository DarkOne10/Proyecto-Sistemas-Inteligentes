from toriel_rl.scenarios.base import BaseScenario


class NarrowCorridorScenario(BaseScenario):
    """D5: a short horizontal corridor where aimed darts have nowhere to hide."""

    name = "narrow_corridor"
    default_pattern = "dart_volley"
    base_size = (192.0, 64.0)
