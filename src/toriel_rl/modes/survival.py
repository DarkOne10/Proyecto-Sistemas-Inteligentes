from toriel_rl.modes.base import BaseMode, StepContext
from toriel_rl.rewards import RewardConfig


class SurvivalMode(BaseMode):
    """B1: survive the gauntlet; rewards clean near-misses while the rain ramps up."""

    name = "survival"
    duration = 30.0
    default_pattern = "rain"

    def reward_config(self) -> RewardConfig:
        return RewardConfig()

    def extra_reward(self, ctx: StepContext) -> float:
        if 0.0 < ctx.nearest_gap < 10.0:
            return self.reward_config().near_miss
        return 0.0
