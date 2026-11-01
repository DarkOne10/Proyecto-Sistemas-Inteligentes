import argparse
import csv
from pathlib import Path

from toriel_rl.agent import QLearningAgent
from toriel_rl.config import EnvConfig
from toriel_rl.environment import TorielEnv


def train_agent(
    episodes: int = 2000,
    seed: int = 7,
    output_dir: str | Path = "results",
    config: EnvConfig | None = None,
    full_combat_episodes: int = 500,
) -> list[dict]:
    if episodes <= 0 or full_combat_episodes < 0:
        raise ValueError("Isolated episodes must be positive and full-combat episodes non-negative.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    env = TorielEnv(config=config)
    agent = QLearningAgent(seed=seed + 1)
    records = []
    attack_names = TorielEnv.available_attacks
    episode_specs = [
        (attack_names[episode % len(attack_names)], True)
        for episode in range(episodes)
    ]
    episode_specs.extend([("full_combat", False)] * full_combat_episodes)
    full_combat_started = False

    try:
        for episode, (attack_name, attack_only) in enumerate(episode_specs):
            if not attack_only and not full_combat_started:
                agent.epsilon = max(agent.epsilon, 0.3)
                full_combat_started = True
            reset_options = (
                {"attack": attack_name, "attack_only": True}
                if attack_only
                else {}
            )
            if attack_only and attack_name == "paw_sweep":
                reset_options["spares"] = 2
            observation, _ = env.reset(seed=seed + episode, options=reset_options)
            total_reward = 0.0
            total_damage = 0
            steps = 0
            terminated = False
            truncated = False
            info = {}

            while not (terminated or truncated):
                action = agent.act(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                agent.learn(
                    observation,
                    action,
                    reward,
                    next_observation,
                    terminated=terminated,
                    truncated=truncated,
                )
                observation = next_observation
                total_reward += reward
                total_damage += info["damage"]
                steps += 1

            agent.decay_exploration()
            records.append(
                {
                    "episode": episode + 1,
                    "attack": attack_name,
                    "reward": round(total_reward, 4),
                    "steps": steps,
                    "hp": info["hp"],
                    "damage": total_damage,
                    "completed": int(info["reason"] in {"attack_completed", "toriel_stopped"}),
                    "epsilon": round(agent.epsilon, 6),
                }
            )
    finally:
        env.close()

    agent.save(output_path / "q_table.json")
    with (output_path / "training.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Toriel Q-learning agent.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--full-combat-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    records = train_agent(
        args.episodes,
        args.seed,
        args.output_dir,
        full_combat_episodes=args.full_combat_episodes,
    )
    recent = records[-min(100, len(records)):]
    completion_rate = sum(row["completed"] for row in recent) / len(recent)
    mean_damage = sum(row["damage"] for row in recent) / len(recent)
    print(f"Total episodes: {len(records)} ({args.episodes} isolated + {args.full_combat_episodes} full combat)")
    print(f"Recent completion rate: {completion_rate:.1%}")
    print(f"Recent mean damage: {mean_damage:.2f} HP")
    print(f"Saved model and log to: {args.output_dir}")


if __name__ == "__main__":
    main()