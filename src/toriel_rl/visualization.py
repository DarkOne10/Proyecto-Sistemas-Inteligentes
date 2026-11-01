from toriel_rl.attacks.faltering import FalteringAttack
from toriel_rl.attacks.fire_helix import FireHelixAttack


def draw_environment(surface, env, pygame) -> None:
    config = env.config
    scale = surface.get_width() / config.board_width
    surface.fill((12, 14, 20))

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
    elif isinstance(env.attack, FalteringAttack):
        for fireball in env.attack.fireballs:
            pygame.draw.circle(
                surface,
                (245, 114, 55),
                (int(fireball.x * scale), int(fireball.y * scale)),
                max(1, int(fireball.radius * scale)),
            )

    soul_x = int(env.soul.x * scale)
    soul_y = int(env.soul.y * scale)
    radius = max(2, int(env.soul.radius * scale))
    pygame.draw.polygon(
        surface,
        (255, 55, 80),
        [
            (soul_x, soul_y - radius),
            (soul_x + radius, soul_y),
            (soul_x, soul_y + radius),
            (soul_x - radius, soul_y),
        ],
    )
    font = pygame.font.Font(None, 24)
    label = font.render(f"HP {env.soul.hp}   {env.attack.name}", True, (245, 245, 245))
    surface.blit(label, (8, 8))