import argparse
import csv
from pathlib import Path

from toriel_rl.agent import QLearningAgent
from toriel_rl.config import EnvConfig
from toriel_rl.environment import TorielEnv


def evaluate_policies(
    agent: QLearningAgent,
    episodes: int = 100,
    seed: int = 10000,
    config: EnvConfig | None = None,
) -> dict[str, dict[str, list[dict]]]:
    if episodes <= 0:
        raise ValueError("Episodes must be positive.")

    results = {
        attack_name: {"random": [], "q_learning": []}
        for attack_name in (*TorielEnv.available_attacks, "full_combat")
    }
    for attack_name, policy_results in results.items():
        for policy_name in policy_results:
            env = TorielEnv(config=config)
            try:
                for episode in range(episodes):
                    episode_seed = seed + episode
                    reset_options = (
                        {}
                        if attack_name == "full_combat"
                        else {"attack": attack_name, "attack_only": True}
                    )
                    if attack_name == "paw_sweep":
                        reset_options["spares"] = 2
                    observation, _ = env.reset(seed=episode_seed, options=reset_options)
                    env.action_space.seed(episode_seed + 1)
                    total_reward = 0.0
                    total_damage = 0
                    steps = 0
                    terminated = False
                    truncated = False
                    info = {}

                    while not (terminated or truncated):
                        if policy_name == "random":
                            action = env.action_space.sample()
                        else:
                            action = agent.act(observation, explore=False)
                        observation, reward, terminated, truncated, info = env.step(action)
                        total_reward += reward
                        total_damage += info["damage"]
                        steps += 1

                    policy_results[policy_name].append(
                        {
                            "episode": episode + 1,
                            "seed": episode_seed,
                            "reward": round(total_reward, 4),
                            "steps": steps,
                            "hp": info["hp"],
                            "damage": total_damage,
                            "completed": int(
                                info["reason"]
                                in ({"toriel_stopped"} if attack_name == "full_combat" else {"attack_completed"})
                            ),
                        }
                    )
            finally:
                env.close()
    return results


def summarize(records: list[dict]) -> dict[str, float]:
    episode_count = len(records)
    if episode_count == 0:
        raise ValueError("Cannot summarize an empty evaluation.")
    return {
        "completion_rate": sum(row["completed"] for row in records) / episode_count,
        "mean_damage": sum(row["damage"] for row in records) / episode_count,
        "mean_hp": sum(row["hp"] for row in records) / episode_count,
        "mean_reward": sum(row["reward"] for row in records) / episode_count,
    }


def save_results(results: dict[str, dict[str, list[dict]]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["attack", "policy", "episode", "seed", "reward", "steps", "hp", "damage", "completed"]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for attack_name, policies in results.items():
            for policy_name, records in policies.items():
                for record in records:
                    writer.writerow({"attack": attack_name, "policy": policy_name, **record})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Q-learning against a random policy.")
    parser.add_argument("--model", type=Path, default=Path("results/q_table.json"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.csv"))
    args = parser.parse_args()

    agent = QLearningAgent.load(args.model)
    results = evaluate_policies(agent, args.episodes, args.seed)
    for attack_name, policies in results.items():
        for policy_name, records in policies.items():
            metrics = summarize(records)
            print(
                f"{attack_name} / {policy_name}: completion={metrics['completion_rate']:.1%}, "
                f"mean damage={metrics['mean_damage']:.2f} HP, "
                f"mean HP={metrics['mean_hp']:.2f}, reward={metrics['mean_reward']:.2f}"
            )
    save_results(results, args.output)
    print(f"Saved evaluation to: {args.output}")


if __name__ == "__main__":
    main()