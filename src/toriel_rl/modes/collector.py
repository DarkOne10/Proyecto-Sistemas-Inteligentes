from toriel_rl.modes.base import BaseMode, StepContext
from toriel_rl.rewards import RewardConfig


class CollectorMode(BaseMode):
    """B2: race through obstacles to grab the green pickup quota before time runs out."""

    name = "collector"
    duration = 25.0
    default_pattern = "pickups"

    def reset(self, rng) -> None:
        self.target = 4 + self.level

    def reward_config(self) -> RewardConfig:
        return RewardConfig()

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        width = self.config.board_width
        height = self.config.board_height
        return [
            (width * 0.25 - 6.0, height * 0.2 - 6.0, 12.0, height * 0.35),
            (width * 0.75 - 6.0, height * 0.45, 12.0, height * 0.35),
        ]

    def completion_reason(self, ctx: StepContext) -> str | None:
        if ctx.greens_total >= self.target:
            return "objective_met"
        return None

    def completion_bonus(self, ctx: StepContext) -> float:
        if ctx.greens_total >= self.target:
            return self.reward_config().objective
        return 0.0
