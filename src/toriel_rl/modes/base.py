from abc import ABC
from dataclasses import dataclass

from toriel_rl.config import EnvConfig
from toriel_rl.entities.fireball import Fireball
from toriel_rl.entities.soul import Soul
from toriel_rl.rewards import RewardConfig


@dataclass(frozen=True)
class StepContext:
    elapsed: float
    soul: Soul
    bullets: tuple[Fireball, ...]
    damage: int
    greens: int
    greens_total: int
    nearest_gap: float
    moving: bool
    focus: bool
    iframes: bool
    whites_only: bool
    level: int
    terminated: bool


class BaseMode(ABC):
    """Objective, reward shaping, duration, and board dynamics for a sandbox run."""

    name = "neutral"
    duration: float | None = None
    default_pattern: str | None = None
    action_mode: str | None = None

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.level = 1
        self.duration_seconds = self.duration or config.episode_seconds

    def reset(self, rng) -> None:
        return None

    def reward_config(self) -> RewardConfig:
        return RewardConfig()

    def extra_reward(self, ctx: StepContext) -> float:
        return 0.0

    def completion_reason(self, ctx: StepContext) -> str | None:
        return None

    def completion_bonus(self, ctx: StepContext) -> float:
        return 0.0

    def board_size(self, elapsed: float) -> tuple[float, float] | None:
        return None

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        return []

    def dynamic_level(self, elapsed: float) -> int | None:
        """Escalate difficulty mid-episode (return None to keep the reset level)."""
        return None

    def select_pattern(self, rng, available: tuple[str, ...]) -> str | None:
        """Re-roll the episode pattern during reset (return None to use the defaults)."""
        return None

    def select_scenario(self, rng, available: tuple[str, ...]) -> str | None:
        """Re-roll the episode scenario during reset (return None for an open board)."""
        return None
