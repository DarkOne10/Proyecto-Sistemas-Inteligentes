from dataclasses import dataclass

from toriel_rl.obs import ObsConfig


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    timesteps: int
    env_kwargs: dict
    min_mean_return: float
    min_completion: float


def default_stages() -> list[CurriculumStage]:
    easy_vision = {"obs_config": ObsConfig(mask_k=3)}
    return [
        CurriculumStage(
            "intro_drifter", 15_000,
            {"mode": "survival", "pattern": "drifter", "level": 1, **easy_vision},
            -25.0, 0.6,
        ),
        CurriculumStage(
            "sine_dodge", 15_000,
            {"mode": "survival", "pattern": "sine_sweeper", "level": 1, **easy_vision},
            -25.0, 0.6,
        ),
        CurriculumStage(
            "gates_rain", 20_000,
            {"mode": "survival", "pattern": "orange_gates", "level": 2},
            -25.0, 0.5,
        ),
        CurriculumStage(
            "spiral_evasion", 20_000,
            {"mode": "survival", "scenario": "spiral_evasion", "level": 2},
            -25.0, 0.5,
        ),
        CurriculumStage(
            "slalom_focus", 25_000,
            {"mode": "focus_precision", "scenario": "slalom_switching", "level": 3},
            -20.0, 0.4,
        ),
    ]


class Curriculum:
    """Five-stage training ladder: easy patterns with masked observations up to full scenarios."""

    def __init__(self, stages: list[CurriculumStage] | None = None) -> None:
        self.stages = default_stages() if stages is None else list(stages)
        if not self.stages:
            raise ValueError("A curriculum needs at least one stage.")
        self.index = 0

    @property
    def current(self) -> CurriculumStage:
        return self.stages[self.index]

    @property
    def finished(self) -> bool:
        return self.index >= len(self.stages)

    @property
    def env_kwargs(self) -> dict:
        return dict(self.current.env_kwargs)

    def should_promote(self, stats: dict) -> bool:
        stage = self.current
        return (
            stats["mean_return"] >= stage.min_mean_return
            and stats["completion_rate"] >= stage.min_completion
        )

    def advance(self) -> None:
        self.index += 1
