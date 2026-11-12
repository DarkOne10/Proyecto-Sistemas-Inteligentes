from toriel_rl.bullets.base import BasePattern


class BaseScenario(BasePattern):
    """Arena configuration layered on top of the bullet sandbox."""

    default_pattern: str | None = None
    base_size: tuple[float, float] | None = None
    level: int = 1

    def reset(self, rng) -> None:
        return None

    def update(self, dt: float, soul_x: float, soul_y: float) -> None:
        return None

    def bullets(self) -> list:
        return []

    def board_size(self, elapsed: float) -> tuple[float, float] | None:
        return self.base_size

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        return []
