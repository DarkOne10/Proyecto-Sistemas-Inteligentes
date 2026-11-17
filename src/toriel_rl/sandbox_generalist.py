"""Generalist training: one PPO policy that dodges any sandbox scene, not just one task.

Trains in the ``gauntlet`` mode (random pattern / scenario / level per episode) with
parallel subprocess environments, evaluates periodically on a fixed held-out suite that
covers every scene of the showcase/arena, and keeps the best checkpoint.

    python -m toriel_rl.sandbox_generalist --timesteps 3000000
    python -m toriel_rl.sandbox_generalist --eval-only models/gauntlet/sandbox_ppo.zip
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toriel_rl.bullets import PATTERN_TYPES
from toriel_rl.config import EnvConfig
from toriel_rl.modes.mixup import TRIVIAL_PATTERNS
from toriel_rl.sandbox_env import SoulSandboxEnv
from toriel_rl.scenarios import SCENARIO_TYPES

EVAL_SEED = 50_000  # never used for training (training seeds start at --seed)
EVAL_SECONDS = 15.0  # same length as a showcase/arena episode
# Roughly log-spaced: most of the visible learning happens in the first few 100k steps.
DEFAULT_CHECKPOINT_STEPS = (25_000, 50_000, 100_000, 200_000, 400_000, 800_000, 1_600_000, 3_000_000)


@dataclass(frozen=True)
class SuiteScene:
    kind: str  # "pattern" | "scenario"
    key: str
    level: int

    @property
    def options(self) -> dict:
        if self.kind == "pattern":
            return {"pattern": self.key, "scenario": None, "level": self.level}
        return {"scenario": self.key, "level": self.level}


def build_suite(levels=(1, 2, 3)) -> list[SuiteScene]:
    scenes = [
        SuiteScene("pattern", name, level)
        for name in PATTERN_TYPES
        if name not in TRIVIAL_PATTERNS
        for level in levels
    ]
    scenes += [SuiteScene("scenario", name, level) for name in SCENARIO_TYPES for level in levels]
    return scenes


def evaluate_suite(act, action_mode: str, *, episodes: int = 1, seed: int = EVAL_SEED, levels=(1, 2, 3)) -> dict:
    """Roll ``act(obs, env)`` over every suite scene; returns per-scene and overall stats."""
    env = SoulSandboxEnv(config=EnvConfig(episode_seconds=EVAL_SECONDS), action_mode=action_mode)
    per_scene: dict[str, dict] = {}
    for index, scene in enumerate(build_suite(levels)):
        for episode in range(episodes):
            episode_seed = seed + index * 101 + episode
            observation, _ = env.reset(seed=episode_seed, options=scene.options)
            env.action_space.seed(episode_seed)
            total, done, info = 0.0, False, {}
            while not done:
                observation, reward, terminated, truncated, info = env.step(act(observation, env))
                total += reward
                done = terminated or truncated
            stats = per_scene.setdefault(scene.key, {"return": [], "hp": [], "survived": []})
            stats["return"].append(total)
            stats["hp"].append(info["hp"])
            stats["survived"].append(info["reason"] != "death")
    env.close()

    summary = {
        key: {
            "return": float(np.mean(v["return"])),
            "hp": float(np.mean(v["hp"])),
            "survival": float(np.mean(v["survived"])),
        }
        for key, v in per_scene.items()
    }
    summary_all = {
        "return": float(np.mean([s["return"] for s in summary.values()])),
        "hp": float(np.mean([s["hp"] for s in summary.values()])),
        "survival": float(np.mean([s["survival"] for s in summary.values()])),
        "worst_scene": min(summary, key=lambda k: summary[k]["survival"]),
    }
    return {"overall": summary_all, "scenes": summary}


def random_actor(observation, env):
    return env.action_space.sample()


def model_actor(model):
    def act(observation, env):
        action, _ = model.predict(observation, deterministic=True)
        return action

    return act


def _make_env(rank: int, seed: int, action_mode: str):
    def thunk():
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(SoulSandboxEnv(mode="gauntlet", action_mode=action_mode))
        env.reset(seed=seed + rank)
        return env

    return thunk


def train_generalist(
    timesteps: int = 3_000_000,
    *,
    seed: int = 7,
    n_envs: int = 32,
    eval_every: int = 250_000,
    output_dir: str | Path = "models/gauntlet",
    action_mode: str = "focus",
    init_model: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_steps=DEFAULT_CHECKPOINT_STEPS,
):
    """Train in gauntlet mode; optionally also save ``pasos_<n>.zip`` snapshots at
    ``checkpoint_steps`` so the learning progress can be replayed later."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # One process with many envs: the policy runs batched and there is no per-step IPC
    # (SubprocVecEnv was ~2x slower here because every step is a pipe round-trip).
    torch.set_num_threads(4)
    vec_env = VecMonitor(DummyVecEnv([_make_env(rank, seed, action_mode) for rank in range(n_envs)]))

    def schedule(progress_remaining: float) -> float:
        return 3e-4 * max(progress_remaining, 0.1)

    if init_model is not None:
        model = PPO.load(init_model, env=vec_env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=512,
            batch_size=2048,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            learning_rate=schedule,
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
            seed=seed,
            device="cpu",
            verbose=0,
        )

    snapshots = sorted(checkpoint_steps) if checkpoint_dir is not None else []
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output / "generalist_curve.csv"
    best = {"score": -np.inf}

    class SuiteEval(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.next_eval = eval_every
            self.started = time.time()

        def _on_step(self) -> bool:
            while snapshots and self.num_timesteps >= snapshots[0]:
                step = snapshots.pop(0)
                self.model.save(checkpoint_dir / f"pasos_{step:08d}.zip")
            if self.num_timesteps < self.next_eval:
                return True
            self.next_eval += eval_every
            self._evaluate()
            return True

        def _evaluate(self) -> None:
            result = evaluate_suite(model_actor(self.model), action_mode)["overall"]
            score = result["survival"] * 100 + result["return"]
            row = {
                "timesteps": self.num_timesteps,
                "minutes": round((time.time() - self.started) / 60, 1),
                **{k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()},
            }
            new_file = not curve_path.exists()
            with curve_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            marker = ""
            if score > best["score"]:
                best["score"] = score
                self.model.save(output / "sandbox_ppo.zip")
                marker = "  <- mejor"
            print(
                f"[gauntlet] {self.num_timesteps:>9,} pasos | {row['minutes']:5.1f} min | "
                f"supervivencia {result['survival'] * 100:5.1f}% | retorno {result['return']:+7.1f} | "
                f"HP {result['hp']:4.1f} | peor escena {result['worst_scene']}{marker}",
                flush=True,
            )

    if curve_path.exists():
        curve_path.unlink()
    callback = SuiteEval()
    model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=init_model is None)
    callback._evaluate()
    model.save(output / "sandbox_ppo_last.zip")
    vec_env.close()
    return output / "sandbox_ppo.zip"


def _print_table(results: dict[str, dict]) -> None:
    names = list(results)
    scenes = list(next(iter(results.values()))["scenes"])
    header = f"{'escena':18}" + "".join(f"{n[:22]:>24}" for n in names)
    print(header)
    print("-" * len(header))
    for scene in scenes:
        cells = "".join(
            f"{results[n]['scenes'][scene]['survival'] * 100:9.0f}% {results[n]['scenes'][scene]['return']:+7.1f}      "
            for n in names
        )
        print(f"{scene:18}{cells}")
    print("-" * len(header))
    cells = "".join(
        f"{results[n]['overall']['survival'] * 100:9.1f}% {results[n]['overall']['return']:+7.1f}      " for n in names
    )
    print(f"{'TOTAL':18}{cells}")
    print("(cada celda: % de episodios sobrevividos, retorno medio)")


def compare(model_paths: list[Path], *, episodes: int = 2, output: Path | None = None) -> dict:
    """Random baseline vs. each saved model on the held-out suite."""
    from toriel_rl.sandbox_training import load_model
    from toriel_rl.showcase import ModelPolicy, detect_algo

    results = {"aleatorio": evaluate_suite(random_actor, "focus", episodes=episodes)}
    for path in model_paths:
        model = load_model(detect_algo(path), path)
        mode = ModelPolicy(model).action_mode
        results[str(path.parent.name or path.stem)] = evaluate_suite(model_actor(model), mode, episodes=episodes)
    _print_table(results)
    if output is not None:
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate one generalist PPO policy for every sandbox scene.")
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=250_000)
    parser.add_argument("--output-dir", type=Path, default=Path("models/gauntlet"))
    parser.add_argument("--init-model", type=Path, help="Continue training from this zip.")
    parser.add_argument("--checkpoint-dir", type=Path, help="Save pasos_<n>.zip snapshots here (for gauntlet_progress).")
    parser.add_argument("--eval-only", type=Path, nargs="+", help="Compare these zips against the random baseline.")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Episodes per scene and level in --eval-only.")
    parser.add_argument("--json", type=Path, help="Write the --eval-only comparison to this file.")
    args = parser.parse_args()

    if args.eval_only:
        compare(args.eval_only, episodes=args.eval_episodes, output=args.json)
        return
    path = train_generalist(
        args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        eval_every=args.eval_every,
        output_dir=args.output_dir,
        init_model=args.init_model,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"Mejor modelo guardado en {path}")


if __name__ == "__main__":
    main()
