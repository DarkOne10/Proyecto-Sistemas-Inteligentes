from dataclasses import dataclass
from math import hypot

import numpy as np

from toriel_rl.entities.fireball import Fireball
from toriel_rl.entities.soul import Soul

BULLET_COLORS = ("white", "blue", "orange", "green")
BASE_FEATURE_COUNT = 18
SLOT_FEATURE_COUNT = 6 + len(BULLET_COLORS)


@dataclass(frozen=True)
class ObsConfig:
    k_bullets: int = 8
    mask_k: int | None = None
    max_bullet_speed: float = 200.0
    max_bullet_radius: float = 12.0

    def __post_init__(self) -> None:
        if self.k_bullets < 1:
            raise ValueError("k_bullets must be positive.")
        if self.mask_k is not None and not 0 <= self.mask_k <= self.k_bullets:
            raise ValueError("mask_k must be between zero and k_bullets.")
        if self.max_bullet_speed <= 0 or self.max_bullet_radius <= 0:
            raise ValueError("Bullet normalizers must be positive.")

    @property
    def fill_limit(self) -> int:
        return self.k_bullets if self.mask_k is None else self.mask_k


def _clip(value: float, low: float, high: float) -> float:
    """Scalar clip; np.clip on Python floats dominated the per-step cost."""
    return low if value < low else high if value > high else value


def observation_size(config: ObsConfig) -> int:
    return BASE_FEATURE_COUNT + config.k_bullets * SLOT_FEATURE_COUNT


def build_observation(
    soul: Soul,
    bullets: list[Fireball],
    *,
    board_width: float,
    board_height: float,
    max_hp: int,
    phase: float,
    iframes: bool,
    moving: bool,
    focus: bool,
    config: ObsConfig,
) -> np.ndarray:
    diagonal = hypot(board_width, board_height)
    values: list[float] = [
        soul.x / board_width,
        soul.y / board_height,
        soul.hp / max_hp,
        _clip(float(phase), 0.0, 1.0),
        float(iframes),
        float(focus),
        float(moving),
        max(soul.y - soul.radius, 0.0) / diagonal,
        max(board_height - soul.y - soul.radius, 0.0) / diagonal,
        max(soul.x - soul.radius, 0.0) / diagonal,
        max(board_width - soul.x - soul.radius, 0.0) / diagonal,
    ]

    threats = [bullet for bullet in bullets if bullet.color != "green"]
    if threats:
        nearest = min(threats, key=lambda bullet: hypot(bullet.x - soul.x, bullet.y - soul.y))
        dx = nearest.x - soul.x
        dy = nearest.y - soul.y
        distance = hypot(dx, dy)
        if distance > 0.0:
            values.extend((min(distance / diagonal, 1.0), dy / distance, dx / distance))
        else:
            values.extend((0.0, 0.0, 0.0))
    else:
        values.extend((1.0, 0.0, 0.0))

    total = max(1, len(bullets))
    values.extend(
        sum(1 for bullet in bullets if bullet.color == color) / total
        for color in BULLET_COLORS
    )

    limit = config.fill_limit
    ordered = sorted(bullets, key=lambda bullet: (bullet.x - soul.x) ** 2 + (bullet.y - soul.y) ** 2)
    for bullet in ordered[:limit]:
        values.extend(
            [
                (bullet.x - soul.x) / diagonal,
                (bullet.y - soul.y) / diagonal,
                _clip(bullet.vx / config.max_bullet_speed, -1.0, 1.0),
                _clip(bullet.vy / config.max_bullet_speed, -1.0, 1.0),
                _clip(bullet.radius / config.max_bullet_radius, 0.0, 1.0),
                *(1.0 if bullet.color == color else 0.0 for color in BULLET_COLORS),
                0.0,
            ]
        )
    empty_slot = [0.0] * (SLOT_FEATURE_COUNT - 1) + [1.0]
    for _ in range(limit - min(len(ordered), limit)):
        values.extend(empty_slot)
    for _ in range(config.k_bullets - limit):
        values.extend([0.0] * SLOT_FEATURE_COUNT)

    return np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)
