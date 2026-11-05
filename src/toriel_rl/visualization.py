from toriel_rl.attacks.faltering import FalteringAttack
from toriel_rl.attacks.fire_helix import FireHelixAttack
from toriel_rl.attacks.fire_waves import FireWavesAttack
from toriel_rl.attacks.paw_sweep import PawSweepAttack

BULLET_COLORS = {
    "white": (255, 255, 255),
    "blue": (80, 130, 255),
    "orange": (255, 150, 40),
    "green": (90, 220, 120),
}


def draw_environment(surface, env, pygame) -> None:
    config = env.config
    scale = surface.get_width() / config.board_width
    board_height = int(config.board_height * scale)
    surface.fill((12, 14, 20))
    pygame.draw.rect(surface, (245, 245, 245), (0, 0, surface.get_width(), board_height), 3)

    if isinstance(env.attack, FireHelixAttack):
        left = env.attack.corridor_left
        right = env.attack.corridor_right
        pygame.draw.rect(
            surface,
            (116, 35, 29),
            (0, 0, int(left * scale), surface.get_height()),
        )
        pygame.draw.rect(
            surface,
            (116, 35, 29),
            (int(right * scale), 0, surface.get_width(), surface.get_height()),
        )
        for y in range(8, config.board_height, 16):
            offset = int((y + env.attack.elapsed * 24) % 16)
            for x in range(offset, int(left), 16):
                pygame.draw.circle(
                    surface,
                    (245, 114, 55),
                    (int(x * scale), int(y * scale)),
                    max(2, int(4 * scale)),
                )
            for x in range(int(right) + offset, config.board_width, 16):
                pygame.draw.circle(
                    surface,
                    (245, 114, 55),
                    (int(x * scale), int(y * scale)),
                    max(2, int(4 * scale)),
                )
    elif isinstance(env.attack, FireWavesAttack):
        safe_width = int(env.attack.safe_width * scale)
        safe_height = int(env.attack.safe_height * scale)
        pygame.draw.rect(surface, (24, 58, 52), (0, 0, safe_width, safe_height))
        pygame.draw.rect(
            surface,
            (24, 58, 52),
            (surface.get_width() - safe_width, 0, safe_width, safe_height),
        )
        pygame.draw.rect(surface, (79, 181, 137), (0, 0, safe_width, safe_height), 2)
        pygame.draw.rect(
            surface,
            (79, 181, 137),
            (surface.get_width() - safe_width, 0, safe_width, safe_height),
            2,
        )
        _draw_fireballs(surface, env.attack.fireballs, scale, pygame)
    elif isinstance(env.attack, PawSweepAttack):
        _draw_fireballs(surface, env.attack.fireballs, scale, pygame)
        for paw in env.attack.paws:
            _draw_paw(surface, paw.x, paw.y, paw.radius, scale, pygame)
    elif isinstance(env.attack, FalteringAttack):
        _draw_fireballs(surface, env.attack.fireballs, scale, pygame)

    soul_x = int(env.soul.x * scale)
    soul_y = int(env.soul.y * scale)
    radius = max(2, int(env.soul.radius * scale))
    _draw_soul(surface, soul_x, soul_y, radius, pygame)
    font = pygame.font.Font(None, 23)
    label = font.render(
        f"HP {env.soul.hp}/{config.max_hp}  {env.attack.name}  Spare {env.spare_count}/{config.max_spares}",
        True,
        (245, 245, 245),
    )
    surface.blit(label, (8, board_height + 5))


def draw_sandbox(surface, env, pygame) -> None:
    config = env.config
    window_scale = surface.get_width() / config.board_width
    board_width = int(env.board_width * window_scale)
    board_height = int(env.board_height * window_scale)
    offset_x = (surface.get_width() - board_width) // 2
    offset_y = (surface.get_height() - 32 - board_height) // 2
    surface.fill((12, 14, 20))
    pygame.draw.rect(
        surface, (245, 245, 245), (offset_x, offset_y, board_width, board_height), 3
    )

    for obstacle in env.obstacles:
        ox, oy, ow, oh = obstacle
        pygame.draw.rect(
            surface,
            (58, 60, 72),
            (
                offset_x + int(ox * window_scale),
                offset_y + int(oy * window_scale),
                max(2, int(ow * window_scale)),
                max(2, int(oh * window_scale)),
            ),
        )

    for bullet in env.all_bullets():
        color = BULLET_COLORS.get(bullet.color, (255, 255, 255))
        center = (
            offset_x + int(bullet.x * window_scale),
            offset_y + int(bullet.y * window_scale),
        )
        radius = max(2, int(bullet.radius * window_scale))
        pygame.draw.circle(surface, color, center, radius)
        if bullet.color == "green":
            pygame.draw.circle(surface, (235, 255, 240), center, max(1, int(radius * 0.45)))

    soul_x = offset_x + int(env.soul.x * window_scale)
    soul_y = offset_y + int(env.soul.y * window_scale)
    radius = max(2, int(env.soul.radius * window_scale))
    if not env.iframes_active or int(env.elapsed * 10.0) % 2 == 0:
        _draw_soul(surface, soul_x, soul_y, radius, pygame)

    font = pygame.font.Font(None, 23)
    flags = " ".join(
        flag
        for flag, active in (("MOV", env.moving), ("IFR", env.iframes_active))
        if active
    )
    label = font.render(
        f"HP {env.soul.hp}/{config.max_hp}  {env.pattern.name}  "
        f"bullets {len(env.all_bullets())}  {flags}",
        True,
        (245, 245, 245),
    )
    surface.blit(label, (8, offset_y + board_height + 5))


def _draw_soul(surface, soul_x, soul_y, radius, pygame) -> None:
    pygame.draw.polygon(
        surface,
        (255, 55, 80),
        [
            (soul_x - radius, soul_y - radius // 3),
            (soul_x - radius, soul_y - radius),
            (soul_x - radius // 2, soul_y - radius),
            (soul_x, soul_y - radius // 2),
            (soul_x + radius // 2, soul_y - radius),
            (soul_x + radius, soul_y - radius),
            (soul_x + radius, soul_y - radius // 3),
            (soul_x, soul_y + radius),
        ],
    )


def _draw_fireballs(surface, fireballs, scale, pygame) -> None:
    for fireball in fireballs:
        pygame.draw.circle(
            surface,
            (255, 111, 53),
            (int(fireball.x * scale), int(fireball.y * scale)),
            max(2, int(fireball.radius * scale)),
        )
        pygame.draw.circle(
            surface,
            (255, 203, 99),
            (int(fireball.x * scale), int(fireball.y * scale)),
            max(1, int(fireball.radius * scale * 0.45)),
        )


def _draw_paw(surface, x, y, radius, scale, pygame) -> None:
    center = (int(x * scale), int(y * scale))
    paw_radius = max(4, int(radius * scale))
    pygame.draw.circle(surface, (235, 231, 220), center, paw_radius)
    for offset_x, offset_y in ((-0.58, -0.65), (-0.2, -0.9), (0.2, -0.9), (0.58, -0.65)):
        pygame.draw.circle(
            surface,
            (235, 231, 220),
            (int(center[0] + offset_x * paw_radius), int(center[1] + offset_y * paw_radius)),
            max(2, int(paw_radius * 0.34)),
        )