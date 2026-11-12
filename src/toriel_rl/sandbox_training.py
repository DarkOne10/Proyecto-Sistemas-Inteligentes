import argparse
import csv
from pathlib import Path

import gymnasium as gym

from toriel_rl.config import EnvConfig
from toriel_rl.curriculum import Curriculum
from toriel_rl.sandbox_env import SoulSandboxEnv

CSV_FIELDS = (
    "episode", "length", "return", "hp", "pattern", "scenario", "mode", "level",
    "greens", "dodged", "focus_steps", "wall_hits", "damage_colors", "reason",
)


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
                    "dodged": info["dodged"],
                    "focus_steps": info["focus_steps"],
                    "wall_hits": info["wall_hits"],
                    "damage_colors": ";".join(
                        f"{color}:{amount}" for color, amount in sorted(info["damage_by_color"].items())
                    ),
                    "reason": info["reason"] or "",
                }
            )
        return observation, reward, terminated, truncated, info


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def train_agent(
    timesteps: int = 20_000,
    seed: int = 7,
    n_envs: int = 4,
    output_dir: str | Path = "results",
    config: EnvConfig | None = None,
    rows: list[dict] | None = None,
    algo: str = "ppo",
    **env_kwargs,
) -> tuple[Path, list[dict]]:
    from stable_baselines3.common.env_util import make_vec_env

    if timesteps <= 0:
        raise ValueError("timesteps must be positive.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = []

    def factory() -> EpisodeRecorder:
        return EpisodeRecorder(SoulSandboxEnv(config=config, **env_kwargs), rows)

    vec_env = make_vec_env(factory, n_envs=n_envs, seed=seed)
    model = make_model(algo, vec_env, seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model_path = output_path / f"sandbox_{algo}.zip"
    model.save(model_path)
    vec_env.close()

    _write_csv(output_path / "sandbox_training.csv", rows)
    return model_path, rows


def make_model(algo: str, vec_env, seed: int):
    from gymnasium import spaces
    from stable_baselines3 import PPO

    if algo == "ppo":
        return PPO("MlpPolicy", vec_env, verbose=0, seed=seed)
    if algo == "sac":
        from stable_baselines3 import SAC

        if not isinstance(vec_env.action_space, spaces.Box):
            raise ValueError("SAC requires action_mode='continuous'.")
        return SAC("MlpPolicy", vec_env, verbose=0, seed=seed, buffer_size=10_000)
    raise ValueError(f"Unknown algorithm: {algo}")


def load_model(algo: str, model_path: str | Path):
    if algo == "sac":
        from stable_baselines3 import SAC

        return SAC.load(model_path)
    from stable_baselines3 import PPO

    return PPO.load(model_path)


def evaluate(
    model_path: str | Path,
    episodes: int = 10,
    seed: int = 10_000,
    algo: str = "ppo",
    **env_kwargs,
) -> dict:
    if episodes <= 0:
        raise ValueError("Episodes must be positive.")

    model = load_model(algo, model_path)
    return evaluate_model(model, episodes, seed, **env_kwargs)


def evaluate_model(model, episodes: int, seed: int, **env_kwargs) -> dict:
    """Roll out an in-memory SB3 model for a few seeded episodes."""
    if episodes <= 0:
        raise ValueError("Episodes must be positive.")

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


def train_curriculum(
    seed: int = 7,
    n_envs: int = 4,
    output_dir: str | Path = "results",
    eval_episodes: int = 5,
    max_rounds: int = 3,
    config: EnvConfig | None = None,
    algo: str = "ppo",
    stages=None,
) -> list[dict]:
    curriculum = Curriculum(stages)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    while not curriculum.finished:
        stage = curriculum.current
        rows: list[dict] = []
        stage_dir = output_path / stage.name
        stats: dict = {}
        promoted = False
        for round_index in range(max_rounds):
            model_path, _ = train_agent(
                timesteps=stage.timesteps,
                seed=seed + round_index,
                n_envs=n_envs,
                output_dir=stage_dir,
                config=config,
                rows=rows,
                algo=algo,
                **stage.env_kwargs,
            )
            stats = evaluate(
                model_path,
                episodes=eval_episodes,
                seed=seed + 1_000 + round_index,
                algo=algo,
                **stage.env_kwargs,
            )
            print(
                f"[curriculum] {stage.name} round {round_index + 1}/{max_rounds}: "
                f"return {stats['mean_return']:.2f} | hp {stats['mean_hp']:.1f} | "
                f"completion {stats['completion_rate']:.2f}"
            )
            if curriculum.should_promote(stats):
                promoted = True
                break
        history.append({"stage": stage.name, "promoted": promoted, **stats})
        curriculum.advance()

    with (output_path / "curriculum.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=("stage", "promoted", "mean_return", "mean_hp", "completion_rate")
        )
        writer.writeheader()
        writer.writerows(history)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO/SAC agent in the Red SOUL sandbox.")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pattern", choices=SoulSandboxEnv.available_patterns)
    parser.add_argument("--scenario")
    parser.add_argument("--mode")
    parser.add_argument("--action-mode", choices=("discrete", "focus", "continuous"))
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--algo", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--eval-only", type=Path, help="Evaluate an existing model zip.")
    parser.add_argument("--curriculum", action="store_true", help="Run the five-stage curriculum.")
    args = parser.parse_args()

    scenario = args.scenario if args.scenario not in (None, "none") else None
    mode = args.mode if args.mode not in (None, "none") else None
    env_kwargs = {
        "pattern": args.pattern,
        "scenario": scenario,
        "mode": mode,
        "level": args.level,
    }
    if args.action_mode:
        env_kwargs["action_mode"] = args.action_mode
    elif args.algo == "sac":
        env_kwargs["action_mode"] = "continuous"

    if args.curriculum:
        if args.algo != "ppo":
            parser.error("--curriculum only supports --algo ppo.")
        history = train_curriculum(
            seed=args.seed,
            n_envs=args.n_envs,
            output_dir=args.output_dir,
            eval_episodes=min(args.eval_episodes, 5),
            max_rounds=args.max_rounds,
        )
        promoted = sum(1 for entry in history if entry["promoted"])
        print(f"Curriculum finished: {promoted}/{len(history)} stages promoted on first pass.")
        return

    if args.eval_only is not None:
        model_path = args.eval_only
    else:
        model_path, rows = train_agent(
            args.timesteps,
            args.seed,
            n_envs=args.n_envs,
            output_dir=args.output_dir,
            algo=args.algo,
            **env_kwargs,
        )
        recent = rows[-min(50, len(rows)):]
        if recent:
            print(
                f"Trained {args.algo.upper()} for {args.timesteps} steps | episodes: {len(rows)} | "
                f"recent mean return: {sum(r['return'] for r in recent) / len(recent):.2f} | "
                f"recent mean HP: {sum(r['hp'] for r in recent) / len(recent):.1f}"
            )

    trained = evaluate(model_path, episodes=args.eval_episodes, algo=args.algo, **env_kwargs)
    random = evaluate_random(episodes=args.eval_episodes, **env_kwargs)
    print(f"Trained: {trained}")
    print(f"Random:  {random}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
