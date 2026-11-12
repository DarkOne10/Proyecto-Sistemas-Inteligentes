from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    board_width: int = 192
    board_height: int = 192
    soul_radius: float = 5.0
    soul_speed: float = 90.0
    max_hp: int = 20
    fire_damage: int = 4
    time_step: float = 1.0 / 30.0
    episode_seconds: float = 4.0
    damage_cooldown: float = 0.5
    iframe_duration: float = 0.6
    green_heal: int = 1
    corridor_width_min: float = 48.0
    corridor_width_max: float = 72.0
    faltering_ball_count: int = 8
    paw_sweep_duration: float = 1.5
    max_spares: int = 12
    fire_waves_hard_probability: float = 0.2

    def __post_init__(self) -> None:
        if self.board_width <= 0 or self.board_height <= 0:
            raise ValueError("Board dimensions must be positive.")
        if self.soul_radius <= 0 or self.soul_speed < 0:
            raise ValueError("Soul radius must be positive and speed non-negative.")
        if self.max_hp <= 0 or self.fire_damage <= 0:
            raise ValueError("HP and fire damage must be positive.")
        if self.time_step <= 0 or self.episode_seconds <= 0:
            raise ValueError("Episode duration and time step must be positive.")
        if self.damage_cooldown < 0:
            raise ValueError("Damage cooldown cannot be negative.")
        if self.iframe_duration < 0:
            raise ValueError("Iframe duration cannot be negative.")
        if self.green_heal <= 0:
            raise ValueError("Green heal amount must be positive.")
        if not 0 < self.corridor_width_min <= self.corridor_width_max:
            raise ValueError("Corridor width range is invalid.")
        if self.corridor_width_max > self.board_width - 2 * self.soul_radius:
            raise ValueError("The corridor must leave room for the soul.")
        if self.faltering_ball_count < 0:
            raise ValueError("Faltering ball count cannot be negative.")
        if self.paw_sweep_duration <= 0 or self.max_spares <= 0:
            raise ValueError("Attack duration and spare limit must be positive.")
        if not 0 <= self.fire_waves_hard_probability <= 1:
            raise ValueError("Hard-mode probability must be between zero and one.")