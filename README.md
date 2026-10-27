# RLSnake

A Snake environment with a pygame viewer and a DQN agent that learns to play it.
Training, resuming, evaluation and plotting scripts are included.

## Setup

Requires `numpy`, `pygame`, `torch` and (for plots) `matplotlib`.
Run everything from the project root.

`config.py` is git-ignored and only selects which setup to run. On a fresh
clone, create it from the template:

```
cp config.example.py config.py
```

`checkpoints/`, `logs/` and `analysis/plots/` hold generated output and are also
git-ignored (use `git add -f <file>` for a result worth keeping).

## Configs: one file per setup

Each module in `configs/` is a complete, self-contained setup: hyperparameters
(including the global `SEED`), the environment settings `ENV` (grid, episode
limit and rewards, see [Environment](#environment)), the run-name labels
`ALGO`, `POLICY`, `NET`, `OBS` and `ARCH`, plus two functions that build every
component:

- `make_env(seed)`: the env and its observation encoder
- `make_agent(env, device, seed)`: network, optimizer, loss, replay buffer,
  policy and agent

`train.py`, `eval.py` and `play.py` never build components themselves; they
call these functions on whichever config `config.py` imports:

```python
# config.py
from configs.dqn_cnn import *
```

| Config | Setup |
| --- | --- |
| `configs/dqn_features.py` | DQN + MLP on the 11 hand-crafted features |
| `configs/dqn_cnn.py` | DQN + CNN on the full board (`PlaneObservation`) |
| `configs/dqn_cnn_ego.py` | DQN + CNN on the board rotated so the head faces up (`EgocentricPlaneObservation`), step penalty 0.1 |

To try a new architecture, copy a config, change what it builds (and its `NAME`),
and import it from `config.py`. Don't override values below the import in
`config.py`: the build functions read their own module's globals. To run a
config without touching `config.py` (e.g. two trainings side by side), set
`RLSNAKE_CONFIG=configs.<module>`.

The seed drives the env, replay buffer, exploration policy, network init and the
python/numpy/torch RNGs. Checkpoints go to `checkpoints/<NAME>/`, so setups never
overwrite each other, and resuming or evaluating a checkpoint under a different
config than the one that trained it fails with a clear error.

### Run names

Every run is named from its config (`src/naming.py`), and logs, checkpoints and
plots all use that name, e.g. `dqn_egp_cnn_ego_95_10k_016032064_sp010`. See
[NAMING.md](NAMING.md) for the full format.

## Usage

**Play manually** (Arrows/WASD to move, R to restart, Esc/Q to quit):

```
python play.py
```

**Train** (asks for a log file name under `./logs`, blank = `train_<run>.csv`,
or pass `--log`):

```
python train.py
python train.py --log train_<run>
```

Weights go to `checkpoints/<NAME>/best_<run>.pt` (best 100-episode average) and
`checkpoints/<NAME>/final_<run>.pt`. A full training state (`agent.state_dict()`:
networks, optimizer, policy, replay buffer, ...) is saved to
`checkpoints/<NAME>/state_<run>.pt` every 50 episodes and on Ctrl+C, so training
can be stopped and continued later:

```
python train.py --resume --log train_<run>      # appends to logs/train_<run>.csv
python train.py --resume path/to/state.pt --log train_<run>
```

Resuming refuses a state saved with different settings, except `MAX_EPISODES`,
which may be raised to extend a run (pass the old state's path, since the run
name changes with the episode count).

**Evaluate** a trained agent (greedy, no exploration), with or without a window:

```
python eval.py                                   # checkpoints/<NAME>/best_<run>.pt, 10 episodes
python eval.py --headless --episodes 500 --log eval_<run>
python eval.py --checkpoint best_<run> final_<run>
```

**Plot** training curves and evaluation results to `analysis/plots/` (the
dashboard is saved as `<log name>.png`, single panels in `<log name>/`):

```
python analysis/plot_training.py --csv train_<run>
python analysis/plot_training.py --csv run_a run_b      # compare runs
python analysis/plot_eval.py --csv eval_<run>
```

**Watch an agent** from your own code:

```python
from src.pygame_view import run_simulated

run_simulated(env, lambda obs: agent.act(obs, explore=False), episodes=5)
```

## Project layout

| File | Responsibility |
| --- | --- |
| `config.example.py` | Template for the git-ignored `config.py`, which selects a setup from `configs/` |
| `configs/*.py` | Complete setups: hyperparameters, `make_env(seed)` and `make_agent(env, device, seed)` |
| `play.py` | Entry point for manual play |
| `train.py` | Training loop with periodic resumable checkpoints and CSV metrics |
| `eval.py` | Evaluate trained checkpoints, optionally with a pygame window |
| `analysis/plot_training.py` | Plot training curves from a training log |
| `analysis/plot_eval.py` | Plot evaluation results from an eval log |
| `tests/test_policy.py` | Smoke test of the selected config: build, act, observe, weights and state round-trips |
| `src/environment.py` | Snake state, movement, food spawning, collisions, rewards, episode termination |
| `src/pygame_view.py` | Render the environment, manual play (`run_manual`), policy playback with CSV metrics (`run_simulated`) |
| `src/metadata.py` | Shared types: `Collision` enum, `Transition` and `TransitionBatch` dataclasses |
| `src/logs.py` | Resolves log file names under `./logs` |
| `src/config_loader.py` | Imports `config.py` (or `$RLSNAKE_CONFIG`) and checks it selects a full setup |
| `src/env_settings.py` | `EnvSettings` and `Rewards` dataclasses that configs use to describe the environment |
| `src/naming.py` | Builds run names from a config and the full settings record saved next to the log |
| `src/agents/base.py` | Abstract `Agent` interface (`act`, `observe`, `learn`, `end_episode`, `metrics`, weights and state checkpointing) |
| `src/agents/dqn_agent.py` | DQN agent: online + target network, Huber loss, periodic hard target updates |
| `src/buffers/base.py` | Abstract `Buffer` interface |
| `src/buffers/replay_buffer.py` | Fixed-capacity numpy ring buffer with seeded uniform sampling and checkpoint support |
| `src/networks/base.py` | Abstract `QNetwork` interface |
| `src/networks/mlp_network.py` | 11 -> 64 -> 32 -> 3 MLP that outputs raw Q-values |
| `src/networks/cnn_network.py` | Two 3x3 conv layers + MLP head for (C, H, W) board planes |
| `src/observations/` | Observation encoders: `FeatureObservation`, `GridObservation`, `PlaneObservation`, `EgocentricPlaneObservation` |
| `src/policies/base.py` | Abstract `Policy` interface (plus `end_episode`, `metrics` and state hooks) |
| `src/policies/greedy_policy.py` | Picks the action with the highest Q-value |
| `src/policies/epsilon_greedy_policy.py` | Random action with probability epsilon (decayed once per episode), otherwise greedy |
| `src/policies/softmax_policy.py` | Samples actions from softmax(Q / temperature) (temperature decayed once per episode) |

## Environment

`SnakeEnv(grid_size, start_size, start_pos, start_horizontal=True, seed=67, max_steps=1000, observation=None, rewards=None)`
or `SnakeEnv.from_settings(EnvSettings(...), seed=..., observation=...)` (what
configs use; the snake starts mid-board) follows the Gymnasium convention: `reset()` returns `(observation, info)` and
`step(action)` returns `(observation, reward, terminated, truncated, info)`.

**Actions** are relative to the snake's heading: `0` forward, `1` turn right, `2` turn left.

**Observation** comes from the `observation` encoder; `env.observation_spec.shape`
gives its shape:

- `FeatureObservation` (default), 11 binary features: 3 danger flags (wall or
  body straight, right, left), 4 heading flags, 4 food-location flags
- `GridObservation`: (H, W) board of `Cell` codes (empty, body, head, food);
  categorical, so don't feed it straight into a conv
- `PlaneObservation`: (4, H+2, W+2) wall/body/head/food planes for a CNN; the
  body plane fades from head to tail
- `EgocentricPlaneObservation`: the same planes rotated so the heading always
  points up, matching the relative actions (square boards only)

**Rewards** come from `Rewards(step, food, death, win)`, set per config through
`ENV = EnvSettings(rewards=Rewards(...))`. Defaults: `-0.01` per step, `+10` for
food, `-10` for dying, and an extra `+10` for filling the board.

**Episode end**: `terminated` on wall/self collision or a full board, `truncated` after
`max_steps` steps. The reason is reported in `info["termination_reason"]`.
