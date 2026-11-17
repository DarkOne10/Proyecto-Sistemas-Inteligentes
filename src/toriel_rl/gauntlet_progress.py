"""Replay the gauntlet training snapshots one after another on the same scene and seed.

Shows how the generalist PPO policy improves with training steps, like the Q-Learning
checkpoint viewer does for the original attacks. Snapshots come from::

    python -m toriel_rl.sandbox_generalist --checkpoint-dir checkpoints/gauntlet

and are replayed with::

    python -m toriel_rl.gauntlet_progress --dir checkpoints/gauntlet --scene aimed_burst

Keys: N = skip to the next snapshot, Esc / closing the window = quit.
"""

import argparse
from pathlib import Path

from toriel_rl.bullets import PATTERN_TYPES
from toriel_rl.config import EnvConfig
from toriel_rl.sandbox_env import SoulSandboxEnv
from toriel_rl.scenarios import SCENARIO_TYPES
from toriel_rl.showcase import SCENE_TEXT, ModelPolicy, RandomPolicy, detect_algo

SCENES = tuple(name for name in PATTERN_TYPES if name not in ("empty", "pickups")) + tuple(SCENARIO_TYPES)


def snapshot_runs(directory: Path, *, include_random: bool = True) -> list[tuple[str, Path | None]]:
    """(label, zip) pairs ordered by training steps; ``None`` means the random policy."""
    paths = sorted(directory.glob("pasos_*.zip"), key=lambda p: int(p.stem.split("_")[1]))
    runs: list[tuple[str, Path | None]] = [("Sin entrenar (aleatorio)", None)] if include_random else []
    runs += [(f"{int(p.stem.split('_')[1]):,} pasos".replace(",", "."), p) for p in paths]
    return runs


def scene_options(scene: str, level: int) -> dict:
    if scene in SCENARIO_TYPES:
        return {"scenario": scene, "level": level}
    return {"pattern": scene, "scenario": None, "level": level}


def _draw_banner(surface, pygame, lines: list[tuple[str, tuple]]) -> None:
    font = pygame.font.Font(None, 24)
    height = 8 + 22 * len(lines)
    banner = pygame.Surface((surface.get_width(), height), pygame.SRCALPHA)
    banner.fill((10, 11, 16, 200))
    surface.blit(banner, (0, 0))
    for index, (text, color) in enumerate(lines):
        surface.blit(font.render(text, True, color), (8, 5 + 22 * index))


def _draw_summary(surface, pygame, scene_title: str, results: list[tuple[str, dict]]) -> None:
    surface.fill((10, 11, 16))
    title = pygame.font.Font(None, 30).render(f"Resumen - {scene_title}", True, (245, 245, 245))
    surface.blit(title, (12, 12))
    font = pygame.font.Font(None, 24)
    y = 50
    for label, info in results:
        survived = info["reason"] != "death"
        color = (130, 220, 140) if survived else (240, 90, 100)
        mark = "sobrevive" if survived else f"muere a los {info['t']:.1f}s"
        surface.blit(font.render(f"{label:<26}", True, (220, 220, 230)), (12, y))
        surface.blit(font.render(f"HP {info['hp']:>2}  {mark}", True, color), (surface.get_width() // 2 + 20, y))
        y += 24
    hint = font.render("Cierra la ventana o pulsa Esc para salir", True, (150, 150, 165))
    surface.blit(hint, (12, surface.get_height() - 30))


def run_progress(
    directory: Path,
    scene: str,
    *,
    level: int = 2,
    seed: int = 50_000,
    seconds: float = 15.0,
    speed: float = 1.0,
    include_random: bool = True,
) -> list[tuple[str, dict]]:
    import pygame

    from toriel_rl.sandbox_training import load_model

    runs = snapshot_runs(directory, include_random=include_random)
    if not runs:
        raise SystemExit(f"No hay instantáneas pasos_*.zip en {directory}")
    env = SoulSandboxEnv(config=EnvConfig(episode_seconds=seconds), render_mode="human", action_mode="focus")
    env.present_frames = False
    scene_title = SCENE_TEXT.get(scene, (scene, ""))[0]
    clock = pygame.time.Clock()
    fps = max(1, round(speed / env.config.time_step))
    results: list[tuple[str, dict]] = []

    for index, (label, path) in enumerate(runs, start=1):
        policy = RandomPolicy() if path is None else ModelPolicy(load_model(detect_algo(path), path))
        observation, info = env.reset(seed=seed, options=scene_options(scene, level))
        env.action_space.seed(seed)
        damage, elapsed, done = 0, 0.0, False
        while not done:
            for event in pygame.event.get() if pygame.get_init() else []:
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    env.close()
                    return results
                if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    done = True
            if done:
                break
            observation, _, terminated, truncated, info = env.step(policy.act(observation, env.action_space))
            damage += info.get("damage", 0)
            elapsed += env.config.time_step
            done = terminated or truncated
            env.render()
            _draw_banner(
                pygame.display.get_surface(),
                pygame,
                [
                    (f"{index}/{len(runs)}  {label}", (255, 220, 120)),
                    (f"{scene_title} (nivel {level})  |  daño {damage}  |  {elapsed:4.1f}s", (200, 205, 220)),
                ],
            )
            pygame.display.flip()
            clock.tick(fps)
        info = {**info, "t": elapsed}
        results.append((label, info))
        status = "sobrevive" if info["reason"] != "death" else f"muere a los {elapsed:.1f}s"
        print(f"{label:>26}: HP={info['hp']:>2}  daño={damage:>2}  {status}", flush=True)
        pygame.time.wait(900)

    surface = pygame.display.get_surface()
    while True:
        _draw_summary(surface, pygame, scene_title, results)
        pygame.display.flip()
        event = pygame.event.wait()
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_n)):
            break
    env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ver cómo mejora el modelo gauntlet según los pasos de entrenamiento.")
    parser.add_argument("--dir", type=Path, default=Path("checkpoints/gauntlet"))
    parser.add_argument("--scene", choices=SCENES, default="aimed_burst")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--speed", type=float, default=1.0, help="2 = doble de rápido")
    parser.add_argument("--no-random", action="store_true", help="no mostrar la política aleatoria al inicio")
    args = parser.parse_args()
    run_progress(
        args.dir,
        args.scene,
        level=args.level,
        seed=args.seed,
        seconds=args.seconds,
        speed=args.speed,
        include_random=not args.no_random,
    )


if __name__ == "__main__":
    main()
