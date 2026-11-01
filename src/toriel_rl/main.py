import argparse
from pathlib import Path

from toriel_rl.agent import QLearningAgent
from toriel_rl.environment import TorielEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Toriel RL combat demo.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--attack", choices=(*TorielEnv.available_attacks, "faltering"))
    parser.add_argument("--attack-only", action="store_true")
    args = parser.parse_args()
    if args.attack_only and args.attack is None:
        parser.error("--attack-only requires --attack.")

    import pygame

    agent = QLearningAgent.load(args.model) if args.model is not None else None
    env = TorielEnv(render_mode="human")
    reset_options = {}
    if args.attack is not None:
        reset_options["attack"] = args.attack
    if args.attack_only:
        reset_options["attack_only"] = True
    observation, info = env.reset(seed=args.seed, options=reset_options)
    policy_name = "Q-Learning" if agent is not None else "random"
    print(f"Seed {info['seed']} | policy: {policy_name} | attack: {info['attack']}")
    env.render()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
        action = agent.act(observation, explore=False) if agent else env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated

    print(
        f"Combat ended: {info['reason']} | HP: {info['hp']} | "
        f"Spares: {info['spares']}"
    )
    env.close()


if __name__ == "__main__":
    main()