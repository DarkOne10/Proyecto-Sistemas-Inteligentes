from dataclasses import dataclass


@dataclass
class Fireball:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 3.0
    lane: int = 0
    phase: float = 0.0
    color: str = "white"