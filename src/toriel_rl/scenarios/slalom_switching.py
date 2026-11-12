import numpy as np

from toriel_rl.entities.fireball import Fireball
from toriel_rl.scenarios.base import BaseScenario

WALL_SPACING = 12.0


class SlalomSwitchingScenario(BaseScenario):
    """D3: descending wall rows with alternating side gaps and alternating orange/blue colors,
    forcing the SOUL to weave while switching between moving and still."""

    name = "slalom_switching"
    default_pattern = "empty"
    base_size = (192.0, 128.0)
    damage = 4

    def __init__(self, config) -> None:
        super().__init__(config)
        self._rng: np.random.Generator | None = None
        self._walls: list[dict] = []
        self._bullets: list[Fireball] = []

    @property
    def fall_speed(self) -> float:
        return 45.0 + 7.0 * self.level

    @property
    def gap_half(self) -> float:
        return max(18.0, 26.0 - 2.0 * self.level)

    @property
    def wall_spacing(self) -> float:
        return max(70.0, 100.0 - 8.0 * self.level)

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._walls = [
            {"y": -40.0, "gap_side": 0, "color": "orange", "bullets": []},
            {"y": -40.0 - self.wall_spacing, "gap_side": 1, "color": "blue", "bullets": []},
        ]
        for wall in self._walls:
            wall["bullets"] = self._make_row(wall)
        self._sync()

    def _make_row(self, wall: dict) -> list[Fireball]:
        width = self.config.board_width
        gap_center = width * (0.28 if wall["gap_side"] == 0 else 0.72)
        row = []
        x = self.config.soul_radius
        while x <= width - self.config.soul_radius:
            if abs(x - gap_center) > self.gap_half:
                row.append(
                    Fireball(
                        x=x,
                        y=wall["y"],
                        vx=0.0,
                        vy=0.0,
                        radius=4.0,
                        color=wall["color"],
                    )
                )
            x += WALL_SPACING
        return row

    def _sync(self) -> None:
        self._bullets = [bullet for wall in self._walls for bullet in wall["bullets"]]

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        height = self.config.board_height
        for wall in self._walls:
            wall["y"] += self.fall_speed * dt
            for bullet in wall["bullets"]:
                bullet.y = wall["y"]
        for wall in self._walls:
            if wall["y"] > height + 40.0:
                others = [other["y"] for other in self._walls if other is not wall]
                wall["y"] = min(others) - self.wall_spacing
                wall["gap_side"] = 1 - wall["gap_side"]
                wall["color"] = "blue" if wall["color"] == "orange" else "orange"
                wall["bullets"] = self._make_row(wall)
        self._sync()

    def bullets(self) -> list[Fireball]:
        return self._bullets
