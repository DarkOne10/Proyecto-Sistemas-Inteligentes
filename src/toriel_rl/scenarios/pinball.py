from toriel_rl.scenarios.base import BaseScenario


class PinballScenario(BaseScenario):
    """D4: four bumper blocks around the arena; bullets thread between them."""

    name = "pinball"
    default_pattern = "drifter"
    damage = 4

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        width = self.config.board_width
        height = self.config.board_height
        return [
            (width * 0.2 - 8.0, height * 0.2 - 8.0, 16.0, 16.0),
            (width * 0.8 - 8.0, height * 0.2 - 8.0, 16.0, 16.0),
            (width * 0.2 - 8.0, height * 0.8 - 8.0, 16.0, 16.0),
            (width * 0.8 - 8.0, height * 0.8 - 8.0, 16.0, 16.0),
        ]
