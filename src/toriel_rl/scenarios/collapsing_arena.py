from toriel_rl.scenarios.base import BaseScenario


class CollapsingArenaScenario(BaseScenario):
    """D6: the board shrinks toward its center as the episode runs out of room."""

    name = "collapsing_arena"
    default_pattern = "rain"
    shrink_duration = 15.0
    min_scale = 0.7

    def board_size(self, elapsed: float) -> tuple[float, float]:
        progress = min(max(elapsed, 0.0) / self.shrink_duration, 1.0)
        scale = 1.0 - (1.0 - self.min_scale) * progress
        return self.base_config.board_width * scale, self.base_config.board_height * scale
