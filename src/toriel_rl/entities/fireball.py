from dataclasses import dataclass


@dataclass
class Fireball:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 3.0