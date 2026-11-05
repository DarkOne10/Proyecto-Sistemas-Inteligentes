import math

from toriel_rl.modes.base import BaseMode, StepContext
from toriel_rl.rewards import RewardConfig


class FocusPrecisionMode(BaseMode):
    """B3: pulsating board with blue rain; rewards tight near-misses crossed in focus (slow) mode."""

    name = "focus_precision"
    duration = 30.0
    default_pattern = "blue_rain"
    action_mode = "focus"

    def reward_config(self) -> RewardConfig:
        return RewardConfig()

    def board_size(self, elapsed: float) -> tuple[float, float]:
        pulse = 1.0 + 0.08 * math.sin(math.tau * elapsed / 4.0)
        return self.config.board_width * pulse, self.config.board_height * pulse

    def extra_reward(self, ctx: StepContext) -> float:
        if ctx.focus and 0.0 < ctx.nearest_gap < 12.0:
            return self.reward_config().focus_near_miss
        return 0.0
