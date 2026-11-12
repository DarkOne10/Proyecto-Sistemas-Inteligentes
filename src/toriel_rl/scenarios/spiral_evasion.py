from toriel_rl.scenarios.base import BaseScenario


class SpiralEvasionScenario(BaseScenario):
    """D2: a smaller square arena where the spiral barrage closes in from every edge."""

    name = "spiral_evasion"
    default_pattern = "spiral_barrage"
    base_size = (176.0, 176.0)
