from dataclasses import dataclass
from math import sqrt

_DIAGONAL_FACTOR = 1.0 / sqrt(2.0)

_MOVEMENT = {
    0: (0.0, 0.0),
    1: (0.0, -1.0),
    2: (0.0, 1.0),
    3: (-1.0, 0.0),
    4: (1.0, 0.0),
    5: (-_DIAGONAL_FACTOR, -_DIAGONAL_FACTOR),
    6: (_DIAGONAL_FACTOR, -_DIAGONAL_FACTOR),
    7: (-_DIAGONAL_FACTOR, _DIAGONAL_FACTOR),
    8: (_DIAGONAL_FACTOR, _DIAGONAL_FACTOR),
}


@dataclass
class Soul:
    x: float
    y: float
    radius: float
    hp: int

    def move(self, action: int, speed: float, dt: float, width: float, height: float) -> None:
        dx, dy = _MOVEMENT[action]
        self.x = min(max(self.x + dx * speed * dt, self.radius), width - self.radius)
        self.y = min(max(self.y + dy * speed * dt, self.radius), height - self.radius)
