import argparse
import csv
from pathlib import Path

import gymnasium as gym

from toriel_rl.config import EnvConfig
from toriel_rl.obs import ObsConfig
from toriel_rl.sandbox_env import SoulSandboxEnv


class EpisodeRecorder(gym.Wrapper):
    """Appends one CSV row per finished episode to a shared list."""

    def __init__(self, env, rows: list[dict]) -> None:
        super().__init__(env)
        self._rows = rows
        self._episode_return = 0.0
        self._episode_length = 0

    def reset(self, **kwargs):
        self._episode_return = 0.0
        self._episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += reward
        self._episode_length += 1
        if terminated or truncated:
            self._rows.append(
                {
                    "episode": len(self._rows) + 1,
                    "length": self._episode_length,
                    "return": round(self._episode_return, 4),
                    "hp": info["hp"],
                    "pattern": info["pattern"],
                    "scenario": info["scenario"] or "",
                    "mode": info["mode"],
                    "level": info["level"],
                    "greens": info["greens_total"],
                    "reason": info["reason"] or "",
                }
            )
        return observation, reward, terminated, truncated, info


def make_env_factory(
    *,
    pattern: str | None,
    scenario: str | None,
    mode: str | None,
    level: int,
    rows: list[dict],
    config: EnvConfig | None = None,
):
    def factory() -> EpisodeRecorder:
        env = SoulSandboxEnv(
            config=config,
            obs_config=ObsConfig(),
            pattern=pattern,
            scenario=scenario,
            mode=mode,
            level=level,
        )
        return EpisodeRecorder(env, rows)

    return factory


def train_agent(
    timesteps: int = 20_000,
    seed: int = 7,
    pattern: str | None = None,
    scenario: str | None = None,
    mode: str | None = None,
    level: int = 1,
    n_envs: int = 4,
    output_dir: str | Path = "results",
    config: EnvConfig | None = None,
) -> tuple[Path, list[dict]]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    if timesteps <= 0:
        raise ValueError("timesteps must be positive.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    vec_env = make_vec_env(
        make_env_factory(
            pattern=pattern, scenario=scenario, mode=mode, level=level, rows=rows, config=config
        ),
        n_envs=n_envs,
        seed=seed,
    )
    model = PPO("MlpPolicy", vec_env, verbose=0, seed=seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model_path = output_path / "sandbox_ppo.zip"
    model.save(model_path)
    vec_env.close()

    with (output_path / "sandbox_training.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return model_path, rows


def evaluate(model_path: str | Path, episodes: int = 10, seed: int = 10_000, **env_kwargs) -> dict:
    from stable_baselines3 import PPO

    if episodes <= 0:
        raise ValueError("Episodes must be positive.")

    model = PPO.load(model_path)
    env = SoulSandboxEnv(**env_kwargs)
    returns = []
    hps = []
    completions = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        done = False
        info = {}
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)
        hps.append(info["hp"])
        completions += int(info["reason"] in {"time_up", "objective_met"})
    env.close()
    return {
        "mean_return": sum(returns) / episodes,
        "mean_hp": sum(hps) / episodes,
        "completion_rate": completions / episodes,
    }


def evaluate_random(episodes: int = 10, seed: int = 10_000, **env_kwargs) -> dict:
    if episodes <= 0:
        raise ValueError("Episodes must be positive.")

    env = SoulSandboxEnv(**env_kwargs)
    returns = []
    hps = []
    completions = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        env.action_space.seed(seed + episode)
        total_reward = 0.0
        done = False
        info = {}
        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)
        hps.append(info["hp"])
        completions += int(info["reason"] in {"time_up", "objective_met"})
    env.close()
    return {
        "mean_return": sum(returns) / episodes,
        "mean_hp": sum(hps) / episodes,
        "completion_rate": completions / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent in the Red SOUL sandbox.")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pattern", choices=SoulSandboxEnv.available_patterns)
    parser.add_argument("--scenario")
    parser.add_argument("--mode")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--eval-only", type=Path, help="Evaluate an existing model zip.")
    args = parser.parse_args()

    scenario = args.scenario if args.scenario not in (None, "none") else None
    mode = args.mode if args.mode not in (None, "none") else None
    env_kwargs = {"pattern": args.pattern, "scenario": scenario, "mode": mode, "level": args.level}

    if args.eval_only is not None:
        model_path = args.eval_only
    else:
        model_path, rows = train_agent(
            args.timesteps,
            args.seed,
            pattern=args.pattern,
            scenario=scenario,
            mode=mode,
            level=args.level,
            n_envs=args.n_envs,
            output_dir=args.output_dir,
        )
        recent = rows[-min(50, len(rows)):]
        print(
            f"Trained PPO for {args.timesteps} steps | episodes: {len(rows)} | "
            f"recent mean return: {sum(r['return'] for r in recent) / len(recent):.2f} | "
            f"recent mean HP: {sum(r['hp'] for r in recent) / len(recent):.1f}"
        )

    trained = evaluate(model_path, episodes=args.eval_episodes, **env_kwargs)
    random = evaluate_random(episodes=args.eval_episodes, **env_kwargs)
    print(f"Trained: {trained}")
    print(f"Random:  {random}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
