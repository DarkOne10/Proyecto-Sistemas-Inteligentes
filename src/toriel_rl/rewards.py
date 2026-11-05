from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    survival: float = 0.02
    damage: float = -10.0
    death: float = -20.0
    green: float = 1.0
    clean_completion: float = 5.0
    near_miss: float = 0.1
    focus_near_miss: float = 0.15
    objective: float = 20.0
