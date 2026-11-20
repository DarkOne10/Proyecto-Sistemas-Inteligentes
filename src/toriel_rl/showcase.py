import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from toriel_rl.agent import QLearningAgent
from toriel_rl.bullets import PATTERN_TYPES
from toriel_rl.config import EnvConfig
from toriel_rl.environment import TorielEnv
from toriel_rl.scenarios import SCENARIO_TYPES
from toriel_rl.sandbox_env import SoulSandboxEnv

SCENE_TEXT: dict[str, tuple[str, str]] = {
    # Original Toriel attacks
    "fire_helix": ("Fire Helix", "Stay inside the corridor"),
    "fire_waves": ("Fire Waves", "Wait out the waves in the top corners"),
    "paw_sweep": ("Paw Sweep", "Dodge the paw and its fire trail"),
    # Sandbox patterns
    "sine_sweeper": ("A1 Sine Sweeper", "White always hurts: track the sine"),
    "blue_ring": ("A3 Blue Collapsing Ring", "Blue: stay still to survive"),
    "orange_gates": ("A4 Orange Gate Walls", "Orange: keep moving to survive"),
    "rain": ("Rain", "White always hurts: keep dodging"),
    "homing_sine": ("A2 Homing Sinusoidal", "White always hurts: out-turn the chase"),
    "dart_volley": ("A9 Dart Volley", "White always hurts: dodge the volleys"),
    "spiral_barrage": ("A5 Spiral Barrage", "White always hurts: orbit the gaps"),
    "drifter": ("Drifter", "White always hurts: weave the drifters"),
    "pickups": ("Green Pickups", "Green heals: grab every pickup"),
    "blue_rain": ("Blue Rain", "Blue: stand still or focus"),
    "cross_fire": ("A6 Cross Fire", "White always hurts: cross the gaps"),
    "ricochet": ("A7 Ricochet", "White always hurts: mind the banks"),
    "aimed_burst": ("A8 Aimed Burst", "White always hurts: dodge the bursts"),
    "zigzag_rain": ("A10 Zigzag Rain", "White always hurts: zigzag the rain"),
    "vortex": ("A11 Vortex", "Blue: stay still and ride the wind"),
    # Sandbox scenarios
    "vertical_lanes": ("D1 Vertical Lanes", "White: follow the rotating safe lane"),
    "spiral_evasion": ("D2 Spiral Evasion", "White: climb out of the closing spiral"),
    "slalom_switching": ("D3 Slalom Switching", "Blue: stay still, orange: keep moving"),
    "pinball": ("D4 Pinball", "White: thread between the bumpers"),
    "narrow_corridor": ("D5 Narrow Corridor", "White always hurts: no room to hide"),
    "collapsing_arena": ("D6 Collapsing Arena", "White: the arena keeps shrinking"),
}

_KIND_LABELS = {
    "attack": "Original Toriel attack",
    "pattern": "Sandbox pattern",
    "scenario": "Sandbox scenario",
}


@dataclass(frozen=True)
class Scene:
    kind: str  # "attack" | "pattern" | "scenario"
    key: str
    title: str
    rule: str


def build_scenes() -> tuple[Scene, ...]:
    """Every playable scene in fixed showcase order: attacks, patterns, scenarios."""
    scenes = [Scene("attack", name, *SCENE_TEXT[name]) for name in TorielEnv.available_attacks]
    scenes += [
        Scene("pattern", name, *SCENE_TEXT[name])
        for name in PATTERN_TYPES
        if name != "empty"
    ]
    scenes += [Scene("scenario", name, *SCENE_TEXT[name]) for name in SCENARIO_TYPES]
    return tuple(scenes)


def scene_options(scene: Scene) -> dict:
    if scene.kind == "attack":
        return {"attack": scene.key, "attack_only": True}
    if scene.kind == "pattern":
        return {"pattern": scene.key}
    return {"scenario": scene.key}


class RandomPolicy:
    """Uniform action sampler: the default showcase behaviour."""

    def act(self, observation, action_space):
        return action_space.sample()


class QTablePolicy:
    """Greedy Q-Learning agent for the twelve-value TorielEnv observation."""

    def __init__(self, agent: QLearningAgent) -> None:
        self.agent = agent

    def act(self, observation, action_space):
        return self.agent.act(observation, explore=False)


class ModelPolicy:
    """Deterministic action from a saved stable-baselines3 model."""

    def __init__(self, model) -> None:
        self.model = model

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return tuple(self.model.observation_space.shape)

    @property
    def action_mode(self) -> str:
        from gymnasium import spaces

        space = self.model.action_space
        if isinstance(space, spaces.Box):
            return "continuous"
        if isinstance(space, spaces.MultiDiscrete):
            return "focus"
        return "discrete"

    def act(self, observation, action_space):
        action, _ = self.model.predict(observation, deterministic=True)
        return action


@dataclass(frozen=True)
class Policies:
    random: RandomPolicy
    attack: object
    sandbox: object
    action_mode: str | None = None
    sandbox_obs: tuple[int, ...] | None = None
    label: str = "random"


def detect_algo(model_path: str | Path) -> str:
    """Read a saved SB3 zip well enough to tell PPO from SAC without loading it."""
    import base64
    import json
    import zipfile

    with zipfile.ZipFile(model_path) as archive:
        data = json.loads(archive.read("data"))
    serialized = data.get("policy_class", {}).get(":serialized:", "")
    if serialized:
        raw = base64.b64decode(serialized)
        if b"stable_baselines3.sac" in raw:
            return "sac"
        if b"stable_baselines3.common" in raw:
            return "ppo"
    return "sac" if "buffer_size" in data else "ppo"


def load_policies(model_path: str | Path | None = None, *, algo: str | None = None) -> Policies:
    """Build per-kind policies: a .json runs the Toriel attacks, a .zip the sandbox."""
    random_policy = RandomPolicy()
    if model_path is None:
        return Policies(random_policy, random_policy, random_policy)

    path = Path(model_path)
    if path.suffix == ".json":
        agent = QLearningAgent.load(path)
        return Policies(
            random_policy,
            QTablePolicy(agent),
            random_policy,
            label=f"q-table {path.name}",
        )

    from toriel_rl.sandbox_training import load_model

    model = ModelPolicy(load_model(algo or detect_algo(path), path))
    return Policies(
        random_policy,
        random_policy,
        model,
        action_mode=model.action_mode,
        sandbox_obs=model.observation_shape,
        label=f"model {path.name}",
    )


def open_envs(scenes, config: EnvConfig, *, action_mode: str | None = None, render_mode: str = "human") -> dict:
    """One window-safe env per family: TorielEnv for attacks, SoulSandboxEnv for the rest."""
    kinds = {scene.kind for scene in scenes}
    if not kinds:
        raise ValueError("At least one scene is required.")
    envs: dict = {}
    if "attack" in kinds:
        envs["attack"] = TorielEnv(config=config, render_mode=render_mode)
    if kinds - {"attack"}:
        envs["sandbox"] = SoulSandboxEnv(
            config=config, render_mode=render_mode, action_mode=action_mode
        )
    for env in envs.values():
        env.present_frames = False  # the showcase draws HUD/title on top, then presents once
    return envs


def resolve_policies(policies: Policies, sandbox_env) -> Policies:
    """Downgrade to random when a model's observation size does not fit the sandbox."""
    if sandbox_env is None or policies.sandbox_obs is None:
        return policies
    shape = tuple(sandbox_env.observation_space.shape)
    if shape == policies.sandbox_obs:
        return policies
    print(
        f"[showcase] model expects observation {policies.sandbox_obs} but the sandbox "
        f"provides {shape}; using the random policy there."
    )
    return replace(policies, sandbox=policies.random, sandbox_obs=None)


def scene_env(envs: dict, scene: Scene):
    return envs["attack"] if scene.kind == "attack" else envs["sandbox"]


def scene_policy(policies: Policies, scene: Scene):
    return policies.attack if scene.kind == "attack" else policies.sandbox


def reset_scene(env, scene: Scene, seed: int):
    observation, info = env.reset(seed=seed, options=scene_options(scene))
    env.action_space.seed(seed)
    return observation, info


def pump_events(pygame) -> bool:
    """Drain the event queue; False once the user closes the window."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True


def _font(pygame, size: int):
    return pygame.font.Font(None, size)


def _fit_text(text: str, max_width: int, pygame, size: int, color: tuple):
    font = _font(pygame, size)
    while size > 20 and font.size(text)[0] > max_width:
        size -= 4
        font = _font(pygame, size)
    return font.render(text, True, color)


def _wrap(text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and font.size(candidate)[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _truncate(text: str, font, max_width: int) -> str:
    while text and font.size(text)[0] > max_width:
        text = text[:-1]
    return text


def draw_title_card(surface, scene: Scene, pygame) -> None:
    """Dim the frame and show the scene title, colour rule, and scene family."""
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    overlay.fill((10, 11, 16, 205))
    surface.blit(overlay, (0, 0))

    center_x = surface.get_width() // 2
    center_y = surface.get_height() // 2
    max_width = surface.get_width() - 48

    kind = _font(pygame, 24).render(_KIND_LABELS[scene.kind], True, (172, 178, 194))
    surface.blit(kind, kind.get_rect(center=(center_x, center_y - 58)))

    title = _fit_text(scene.title, max_width, pygame, 50, (245, 245, 245))
    surface.blit(title, title.get_rect(center=(center_x, center_y - 14)))

    rule_font = _font(pygame, 26)
    for index, line in enumerate(_wrap(scene.rule, rule_font, max_width)):
        text = rule_font.render(line, True, (255, 214, 130))
        surface.blit(text, text.get_rect(center=(center_x, center_y + 32 + index * 26)))


def draw_arena_hud(surface, env, scene: Scene, info: dict, reward: float, pygame) -> None:
    """Bottom bar with HP gauge, scene name, episode reward, and iFrame badge."""
    width = surface.get_width()
    height = surface.get_height()
    bar = pygame.Rect(0, height - 32, width, 32)
    pygame.draw.rect(surface, (16, 18, 26), bar)
    pygame.draw.rect(surface, (74, 78, 92), bar, 1)
    font = _font(pygame, 23)

    hp = int(info.get("hp", 0))
    gauge = pygame.Rect(8, height - 21, 84, 10)
    pygame.draw.rect(surface, (74, 26, 34), gauge)
    filled = int(round(gauge.width * hp / max(1, env.config.max_hp)))
    if filled:
        pygame.draw.rect(surface, (240, 74, 96), (gauge.x, gauge.y, filled, gauge.height))
    hp_text = font.render(f"{hp}/{env.config.max_hp}", True, (245, 245, 245))
    surface.blit(hp_text, (gauge.right + 6, height - 24))

    iframes = bool(info.get("iframes"))
    if iframes:
        badge = font.render("IFR", True, (255, 214, 130))
        surface.blit(badge, (width - 8 - badge.get_width(), height - 24))

    reward_text = font.render(f"{reward:+.1f}", True, (168, 214, 255))
    reward_x = width - 8 - reward_text.get_width() - (36 if iframes else 0)
    surface.blit(reward_text, (reward_x, height - 24))

    name_x = gauge.right + 6 + hp_text.get_width() + 12
    name = _truncate(scene.title, font, reward_x - name_x - 8)
    if name:
        surface.blit(font.render(name, True, (245, 245, 245)), (name_x, height - 24))


def _show_scene(env, scene: Scene, policy, *, seed: int, seconds: float, title_seconds: float):
    """Run one scene for a fixed budget; returns its key, or None if the window closed."""
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
    for _ in range(max(1, round(seconds / dt))):
        if not pump_events(pygame):
            return None
        action = policy.act(observation, env.action_space)
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        env.render()
        draw_arena_hud(pygame.display.get_surface(), env, scene, info, total, pygame)
        env.present()
        if terminated or truncated:
            observation, info = reset_scene(env, scene, seed)
            total = 0.0
    return scene.key


def run_showcase(
    model_path: str | Path | None = None,
    *,
    seconds: float = 15.0,
    title_seconds: float = 2.0,
    seed: int = 7,
    loop: bool = False,
    algo: str | None = None,
    scenes=None,
) -> tuple[str, ...]:
    """Cycle through every scene with a title card, then seconds of action."""
    if seconds <= 0:
        raise ValueError("seconds must be positive.")
    if title_seconds < 0:
        raise ValueError("title_seconds cannot be negative.")
    scenes = build_scenes() if scenes is None else tuple(scenes)

    import pygame

    config = EnvConfig(episode_seconds=seconds)
    policies = load_policies(model_path, algo=algo)
    envs = open_envs(scenes, config, action_mode=policies.action_mode)
    policies = resolve_policies(policies, envs.get("sandbox"))
    print(f"[showcase] {len(scenes)} scenes | policy: {policies.label} | {seconds:g}s each")

    shown: list[str] = []
    try:
        while True:
            for index, scene in enumerate(scenes):
                print(f"[showcase] {index + 1}/{len(scenes)} {scene.title} — {scene.rule}")
                key = _show_scene(
                    scene_env(envs, scene),
                    scene,
                    scene_policy(policies, scene),
                    seed=seed + index,
                    seconds=seconds,
                    title_seconds=title_seconds,
                )
                if key is None:
                    return tuple(shown)
                shown.append(key)
            if not loop:
                return tuple(shown)
    finally:
        for env in envs.values():
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Showcase: cycle through every attack, pattern, and scenario with title cards."
    )
    parser.add_argument("--model", type=Path, help="results/q_table.json or a sandbox zip.")
    parser.add_argument("--algo", choices=("ppo", "sac"), help="Override zip algorithm detection.")
    parser.add_argument("--seconds", type=float, default=15.0, help="Action seconds per scene.")
    parser.add_argument("--title-seconds", type=float, default=2.0, help="Title card seconds per scene.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--loop", action="store_true", help="Restart after the last scene.")
    args = parser.parse_args()

    run_showcase(
        args.model,
        seconds=args.seconds,
        title_seconds=args.title_seconds,
        seed=args.seed,
        loop=args.loop,
        algo=args.algo,
    )


if __name__ == "__main__":
    main()
