import argparse
from pathlib import Path

import numpy as np

from toriel_rl.config import EnvConfig
from toriel_rl.showcase import (
    Scene,
    build_scenes,
    draw_arena_hud,
    draw_title_card,
    load_policies,
    open_envs,
    pump_events,
    reset_scene,
    resolve_policies,
    scene_env,
    scene_policy,
)


def _run_episode(env, scene: Scene, policy, *, seed: int, seconds: float, title_seconds: float):
    """Play one episode to its end; returns the record, or None if the window closed."""
    import pygame

    observation, info = reset_scene(env, scene, seed)
    dt = env.config.time_step
    env.render()
    if title_seconds > 0:
        for _ in range(round(title_seconds / dt)):
            if not pump_events(pygame):
                return None
            env.render()
            draw_title_card(pygame.display.get_surface(), scene, pygame)
            env.present()

    total = 0.0
    while True:
        if not pump_events(pygame):
            return None
        action = policy.act(observation, env.action_space)
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        env.render()
        draw_arena_hud(pygame.display.get_surface(), env, scene, info, total, pygame)
        env.present()
        if terminated or truncated:
            break

    record = {
        "scene": scene.key,
        "return": round(total, 2),
        "hp": int(info["hp"]),
        "reason": info["reason"],
    }
    print(
        f"[arena] {scene.title} | return {total:+.1f} | hp {info['hp']} | "
        f"{info['reason'] or 'step limit'}"
    )
    return record


def run_arena(
    model_path: str | Path | None = None,
    *,
    random: bool = False,
    seconds: float = 15.0,
    title_seconds: float = 1.0,
    seed: int = 7,
    episodes: int | None = None,
    algo: str | None = None,
    scenes=None,
) -> tuple[dict, ...]:
    """Roll a random scene per episode until the window closes (or episodes runs out)."""
    if seconds <= 0:
        raise ValueError("seconds must be positive.")
    if title_seconds < 0:
        raise ValueError("title_seconds cannot be negative.")
    if episodes is not None and episodes <= 0:
        raise ValueError("episodes must be positive.")
    scenes = build_scenes() if scenes is None else tuple(scenes)
    if not scenes:
        raise ValueError("At least one scene is required.")

    import pygame

    config = EnvConfig(episode_seconds=seconds)
    policies = load_policies(None if random else model_path, algo=algo)
    envs = open_envs(scenes, config, action_mode=policies.action_mode)
    policies = resolve_policies(policies, envs.get("sandbox"))
    print(f"[arena] policy: {policies.label} | {seconds:g}s per episode")

    rng = np.random.default_rng(seed)
    records: list[dict] = []
    try:
        episode = 0
        while episodes is None or episode < episodes:
            scene = scenes[int(rng.integers(len(scenes)))]
            print(f"[arena] episode {episode + 1}: {scene.title} — {scene.rule}")
            record = _run_episode(
                scene_env(envs, scene),
                scene,
                scene_policy(policies, scene),
                seed=seed + episode,
                seconds=seconds,
                title_seconds=title_seconds,
            )
            if record is None:
                break
            records.append(record)
            episode += 1
        return tuple(records)
    finally:
        for env in envs.values():
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live arena: a random attack, pattern, or scenario on every episode."
    )
    parser.add_argument("--model", type=Path, help="results/q_table.json or a sandbox zip.")
    parser.add_argument("--random", action="store_true", help="Random policy instead of a model.")
    parser.add_argument("--algo", choices=("ppo", "sac"), help="Override zip algorithm detection.")
    parser.add_argument("--episodes", type=int, help="Stop after this many episodes (default: forever).")
    parser.add_argument("--seconds", type=float, default=15.0, help="Seconds per episode.")
    parser.add_argument("--title-seconds", type=float, default=1.0, help="Title card seconds per episode.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.random and args.model is not None:
        parser.error("--random cannot be combined with --model.")
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive.")

    run_arena(
        None if args.random else args.model,
        random=args.random,
        seconds=args.seconds,
        title_seconds=args.title_seconds,
        seed=args.seed,
        episodes=args.episodes,
        algo=args.algo,
    )


if __name__ == "__main__":
    main()
