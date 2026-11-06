import argparse
import csv
from pathlib import Path

from toriel_rl.modes import action_mode_for
from toriel_rl.sandbox_env import SoulSandboxEnv
from toriel_rl.sandbox_training import evaluate_model, evaluate_random, make_model

BENCHMARK_FIELDS = (
    "task",
    "algo",
    "timesteps",
    "mean_return",
    "mean_hp",
    "completion_rate",
    "seed",
)

DEFAULT_TASKS: dict[str, dict] = {
    "survival_rain": {"mode": "survival"},
    "endurance": {"mode": "endurance"},
    "mixup": {"mode": "mixup"},
    "spiral_evasion": {"mode": "survival", "scenario": "spiral_evasion"},
    "collapsing_arena": {"mode": "survival", "scenario": "collapsing_arena"},
    "slalom_focus": {"mode": "focus_precision", "scenario": "slalom_switching"},
}


def _run_kwargs(algo: str, env_kwargs: dict) -> dict | None:
    """SAC always runs continuous; focus-only modes cannot be mapped, so they return None."""
    if algo != "sac":
        return dict(env_kwargs)
    if action_mode_for(env_kwargs.get("mode")) == "focus":
        return None
    return {**env_kwargs, "action_mode": "continuous"}


def benchmark(
    tasks: dict[str, dict] | None = None,
    algos: tuple[str, ...] | list[str] = ("ppo", "sac"),
    timesteps: int = 10_240,
    eval_every: int = 2_048,
    eval_episodes: int = 5,
    seed: int = 7,
    n_envs: int = 1,
    output_dir: str | Path = "results/benchmark",
) -> list[dict]:
    if timesteps <= 0 or eval_every <= 0:
        raise ValueError("timesteps and eval_every must be positive.")
    if eval_episodes <= 0:
        raise ValueError("Episodes must be positive.")

    tasks = DEFAULT_TASKS if tasks is None else tasks
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for task_name, env_kwargs in tasks.items():
        random_stats = evaluate_random(
            episodes=eval_episodes, seed=seed + 90_000, **env_kwargs
        )
        rows.append(
            {
                "task": task_name,
                "algo": "random",
                "timesteps": 0,
                "seed": seed,
                **random_stats,
            }
        )
        for algo in algos:
            run_kwargs = _run_kwargs(algo, env_kwargs)
            if run_kwargs is None:
                print(f"[benchmark] {task_name}/{algo}: skipped (mode requires focus)")
                continue
            rows.extend(
                _run_curve(
                    task_name,
                    algo,
                    run_kwargs,
                    timesteps,
                    eval_every,
                    eval_episodes,
                    seed,
                    n_envs,
                    output_path,
                )
            )

    _write_csv(output_path / "benchmark_curves.csv", rows)
    return rows


def _run_curve(
    task_name: str,
    algo: str,
    env_kwargs: dict,
    timesteps: int,
    eval_every: int,
    eval_episodes: int,
    seed: int,
    n_envs: int,
    output_path: Path,
) -> list[dict]:
    from stable_baselines3.common.env_util import make_vec_env

    def factory() -> SoulSandboxEnv:
        return SoulSandboxEnv(**env_kwargs)

    vec_env = make_vec_env(factory, n_envs=n_envs, seed=seed)
    model = make_model(algo, vec_env, seed)
    curve: list[dict] = []

    while model.num_timesteps < timesteps:
        # learn() adds to num_timesteps, so hand it the remaining slice, not the target.
        chunk = min(eval_every, timesteps - model.num_timesteps)
        model.learn(
            total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False
        )
        stats = evaluate_model(model, eval_episodes, seed + 50_000, **env_kwargs)
        curve.append(
            {
                "task": task_name,
                "algo": algo,
                "timesteps": model.num_timesteps,
                "seed": seed,
                **stats,
            }
        )
        print(
            f"[benchmark] {task_name}/{algo} @{model.num_timesteps}: "
            f"return {stats['mean_return']:.2f} | hp {stats['mean_hp']:.1f} | "
            f"completion {stats['completion_rate']:.2f}"
        )

    vec_env.close()
    model.save(output_path / f"{task_name}_{algo}.zip")
    return curve


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: list[dict], path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(dict.fromkeys(row["task"] for row in rows))
    fig, axes = plt.subplots(
        len(tasks), 1, figsize=(8, 2.8 * len(tasks)), squeeze=False
    )
    for axis, task in zip(axes.flat, tasks):
        for algo in ("random", "ppo", "sac"):
            points = [
                row for row in rows if row["task"] == task and row["algo"] == algo
            ]
            if not points:
                continue
            axis.plot(
                [point["timesteps"] for point in points],
                [point["mean_return"] for point in points],
                marker="o",
                label=algo,
            )
        axis.set_title(task)
        axis.set_xlabel("timesteps")
        axis.set_ylabel("mean return")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    output = Path(path)
    fig.savefig(output)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PPO/SAC across sandbox tasks with periodic evaluation."
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=sorted(DEFAULT_TASKS), default=sorted(DEFAULT_TASKS)
    )
    parser.add_argument(
        "--algos", nargs="+", choices=("ppo", "sac"), default=["ppo", "sac"]
    )
    parser.add_argument("--timesteps", type=int, default=10_240)
    parser.add_argument("--eval-every", type=int, default=2_048)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--plot", action="store_true", help="Save a learning-curve PNG.")
    args = parser.parse_args()

    rows = benchmark(
        tasks={name: DEFAULT_TASKS[name] for name in args.tasks},
        algos=args.algos,
        timesteps=args.timesteps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        n_envs=args.n_envs,
        output_dir=args.output_dir,
    )
    csv_path = args.output_dir / "benchmark_curves.csv"
    print(f"Saved {len(rows)} curve points to {csv_path}")

    if args.plot:
        try:
            png_path = plot_curves(rows, args.output_dir / "benchmark_curves.png")
        except ImportError:
            print("matplotlib is not installed: run pip install -e '.[benchmark]'.")
        else:
            print(f"Saved learning curves to {png_path}")


if __name__ == "__main__":
    main()
