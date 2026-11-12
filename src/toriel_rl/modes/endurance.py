from toriel_rl.modes.base import BaseMode, StepContext
from toriel_rl.rewards import RewardConfig

MILESTONE_SECONDS = 10.0
MILESTONE_REWARD = 2.0


class EnduranceMode(BaseMode):
    """B4: escalating gauntlet — level rises every 10 s and each milestone pays out."""

    name = "endurance"
    duration = 45.0
    default_pattern = "rain"

    def reset(self, rng) -> None:
        self._milestone = 0

    def dynamic_level(self, elapsed: float) -> int:
        return min(5, 1 + int(elapsed // MILESTONE_SECONDS))

    def extra_reward(self, ctx: StepContext) -> float:
        milestone = int(ctx.elapsed // MILESTONE_SECONDS)
        if milestone > self._milestone:
            self._milestone = milestone
            return MILESTONE_REWARD
        return 0.0
