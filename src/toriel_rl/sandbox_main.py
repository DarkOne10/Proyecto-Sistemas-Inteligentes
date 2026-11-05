import argparse

from toriel_rl.config import EnvConfig
from toriel_rl.sandbox_env import SoulSandboxEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Red SOUL sandbox demo.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pattern", choices=SoulSandboxEnv.available_patterns)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--steps", type=int, help="Stop after this many frames.")
    args = parser.parse_args()

    import pygame

    config = EnvConfig(episode_seconds=args.seconds)
    env = SoulSandboxEnv(config=config, render_mode="human")
    reset_options = {"pattern": args.pattern} if args.pattern else {}
    observation, info = env.reset(seed=args.seed, options=reset_options)
    print(f"Seed {info['seed']} | policy: random | pattern: {info['pattern']}")

    env.render()
    greens = 0
    frames = 0
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        greens += info["greens"]
        env.render()
        done = terminated or truncated
        frames += 1
        if args.steps is not None and frames >= args.steps:
            break

    print(
        f"Episode ended: {info['reason'] or 'step limit'} | HP: {info['hp']} | "
        f"greens: {greens}"
    )
    env.close()


if __name__ == "__main__":
    main()
