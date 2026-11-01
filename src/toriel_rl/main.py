import argparse

from toriel_rl.environment import TorielEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a random-policy Toriel RL episode.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import pygame

    env = TorielEnv(render_mode="human")
    _, info = env.reset(seed=args.seed)
    print(f"Seed {info['seed']} | Fire Helix corridor center: {info['corridor_center']:.1f}")
    env.render()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()
        done = terminated or truncated

    print(f"Episode ended: {info['reason']} | HP: {info['hp']}")
    env.close()


if __name__ == "__main__":
    main()