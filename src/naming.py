"""Run names built from the selected config, and the full settings record.

    {algo}_{policy}_{net}_{obs}_{gamma}_{episodes}_{arch}[_{deviation}...]

    dqn_egp_mlp_feat_9_5k_064032
    dqn_egp_cnn_ego_95_10k_016032064_sp010

gamma keeps only the digits after the decimal point (0.9 -> 9, 0.95 -> 95),
episodes are in thousands (10000 -> 10k), and arch is 3 digits per layer.
Deviations list the EnvSettings fields that differ from the defaults:

    sp010  step penalty 0.10        sr005  positive step reward 0.05
    fr20   food reward 20           dr5    death penalty 5
    wb50   win bonus 50             ms500  max steps 500 (msnone = no limit)
    gs8x8  grid size                ss4    start size

Files are then  train_<run>.csv / .json,  eval_<run>.csv  (logs/),
{best,final,state}_<run>.pt  (checkpoints/<NAME>/)  and  <log name>.png
(analysis/plots/). Hyperparameters outside the name (learning rate, epsilon
schedule, ...) are recorded in the train_<run>.json next to the log.
"""

from __future__ import annotations

import dataclasses
import json
import sys

from .env_settings import EnvSettings


def _num(value: float) -> str:
    """1.0 -> '1', 2.5 -> '2p5' (no dots or minus signs in file names)."""
    return f"{value:g}".replace(".", "p").replace("-", "m")


def gamma_tag(gamma: float) -> str:
    if not 0 < gamma < 1:
        raise ValueError(f"gamma must be in (0, 1) to be named, got {gamma}")
    return repr(float(gamma)).split(".")[1]


def episodes_tag(episodes: int) -> str:
    return f"{episodes // 1000}k" if episodes % 1000 == 0 else str(episodes)


def arch_tag(arch) -> str:
    return "".join(f"{size:03d}" for size in arch)


def env_deviations(env: EnvSettings) -> list[str]:
    """One token per EnvSettings field that differs from the default."""
    base = EnvSettings()
    tokens = []

    if env.grid_size != base.grid_size:
        tokens.append("gs{}x{}".format(*env.grid_size))
    if env.start_size != base.start_size:
        tokens.append(f"ss{env.start_size}")
    if env.max_steps != base.max_steps:
        tokens.append("msnone" if env.max_steps is None else f"ms{env.max_steps}")

    rewards, base_rewards = env.rewards, base.rewards
    if rewards.step != base_rewards.step:
        cents = abs(rewards.step) * 100
        amount = f"{round(cents):03d}" if cents == round(cents) else _num(cents)
        tokens.append(("sp" if rewards.step < 0 else "sr") + amount)
    if rewards.food != base_rewards.food:
        tokens.append(f"fr{_num(rewards.food)}")
    if rewards.death != base_rewards.death:
        tokens.append(f"dr{_num(-rewards.death)}")
    if rewards.win != base_rewards.win:
        tokens.append(f"wb{_num(rewards.win)}")

    return tokens


def run_name(config) -> str:
    """The name shared by a run's logs, checkpoints and plots."""
    labels = [config.ALGO, config.POLICY, config.NET, config.OBS]
    for label in labels:
        if not label or "_" in label:
            raise ValueError(f"name labels must be non-empty and contain no '_': {label!r}")

    parts = [
        *labels,
        gamma_tag(config.GAMMA),
        episodes_tag(config.MAX_EPISODES),
        arch_tag(config.ARCH),
        *env_deviations(config.ENV),
    ]
    return "_".join(parts)


def run_settings(config) -> dict:
    """Every UPPER_CASE setting of the selected config module, JSON-ready.

    config.py only star-imports the names in a config's __all__, so read the
    module that defines make_env to get all of its hyperparameters.
    """
    module = sys.modules[config.make_env.__module__]
    settings = {"config_module": module.__name__}

    for key, value in vars(module).items():
        # Skip imported classes that happen to be upper case (MLP, CNN).
        if not key.isupper() or callable(value):
            continue
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        settings[key] = value

    # Round-trip so tuples become lists, i.e. exactly what a saved file holds.
    return json.loads(json.dumps(settings, default=str))
