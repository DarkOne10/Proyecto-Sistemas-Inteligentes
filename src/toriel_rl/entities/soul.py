from dataclasses import dataclass


@dataclass
class Soul:
    x: float
    y: float
    radius: float
    hp: int

    def move(self, action: int, speed: float, dt: float, width: float, height: float) -> None:
        movement = {
            0: (0.0, 0.0),
            1: (0.0, -1.0),
            2: (0.0, 1.0),
            3: (-1.0, 0.0),
            4: (1.0, 0.0),
        }
        dx, dy = movement[action]
        self.x = min(max(self.x + dx * speed * dt, self.radius), width - self.radius)
        self.y = min(max(self.y + dy * speed * dt, self.radius), height - self.radius)