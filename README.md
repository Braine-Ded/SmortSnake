# RLSnake

A Snake environment with a pygame viewer and a DQN agent that learns to play it.
Training, resuming, evaluation and plotting scripts are included.

## Setup

Requires `numpy`, `pygame`, `torch` and (for plots) `matplotlib`.
Run everything from the project root.

`config.py` is git-ignored so per-run hyperparameters don't clutter history.
On a fresh clone, create it from the template:

```
cp config.example.py config.py
```

`checkpoints/`, `logs/` and `analysis/plots/` hold generated output and are also
git-ignored (use `git add -f <file>` for a result worth keeping).

All hyperparameters, including the global `SEED`, live in `config.py`.
The seed drives the env, replay buffer, exploration policy, network init and the
python/numpy/torch RNGs.

## Usage

**Play manually** (Arrows/WASD to move, R to restart, Esc/Q to quit):

```
python play.py
```

**Train** (asks for a log file name under `./logs`, or pass `--log`):

```
python train.py
python train.py --log run1
```

Weights go to `checkpoints/best.pt` (best 100-episode average) and
`checkpoints/final.pt`. A full training state (model, target network, optimizer,
epsilon, replay buffer, ...) is saved to `checkpoints/state.pt` every 50 episodes
and on Ctrl+C, so training can be stopped and continued later:

```
python train.py --resume --log run1      # appends to logs/run1.csv
python train.py --resume path/to/state.pt --log run1
```

**Evaluate** a trained agent (greedy, no exploration), with or without a window:

```
python eval.py                                   # best.pt, 10 episodes
python eval.py --headless --episodes 500
python eval.py --checkpoint checkpoints/best.pt checkpoints/final.pt
```

**Plot** training curves and evaluation results to `analysis/plots/`:

```
python analysis/plot_training.py --csv run1
python analysis/plot_training.py --csv run_a run_b      # compare runs
python analysis/plot_eval.py --csv eval
```

**Watch a policy** from your own code:

```python
from src.pygame_view import run_simulated
from src.policies import GreedyPolicy

policy = GreedyPolicy(model, device="cpu")
run_simulated(env, policy.select_action, episodes=5)
```

## Project layout

| File | Responsibility |
| --- | --- |
| `config.example.py` | Template for the git-ignored `config.py`: seed, board size and hyperparameters (learning rate, gamma, epsilon schedule, batch size, replay capacity, target update frequency, max episodes) |
| `play.py` | Entry point for manual play |
| `train.py` | Training loop with periodic resumable checkpoints and CSV metrics |
| `eval.py` | Evaluate trained checkpoints, optionally with a pygame window |
| `analysis/plot_training.py` | Plot training curves from a training log |
| `analysis/plot_eval.py` | Plot evaluation results from an eval log |
| `tests/test_policy.py` | Smoke test wiring the network, policy, environment and buffer together |
| `src/environment.py` | Snake state, movement, food spawning, collisions, rewards, episode termination |
| `src/pygame_view.py` | Render the environment, manual play (`run_manual`), policy playback with CSV metrics (`run_simulated`) |
| `src/metadata.py` | Shared types: `Collision` enum, `Transition` and `TransitionBatch` dataclasses |
| `src/logs.py` | Resolves log file names under `./logs` |
| `src/agents/base.py` | Abstract `Agent` interface (`act`, `observe`, `learn`) |
| `src/agents/dqn_agent.py` | DQN agent: online + target network, Huber loss, periodic hard target updates |
| `src/buffers/base.py` | Abstract `Buffer` interface |
| `src/buffers/replay_buffer.py` | Fixed-capacity numpy ring buffer with seeded uniform sampling and checkpoint support |
| `src/networks/base.py` | Abstract `QNetwork` interface |
| `src/networks/mlp_network.py` | 11 -> 64 -> 32 -> 3 MLP that outputs raw Q-values |
| `src/policies/base.py` | Abstract `Policy` interface |
| `src/policies/greedy_policy.py` | Picks the action with the highest Q-value |
| `src/policies/epsilon_greedy_policy.py` | Random action with probability epsilon (decayed once per episode), otherwise greedy |

## Environment

`SnakeEnv(grid_size, start_size, start_pos, start_horizontal=True, seed=67, max_steps=1000)`
follows the Gymnasium convention: `reset()` returns `(observation, info)` and
`step(action)` returns `(observation, reward, terminated, truncated, info)`.

**Actions** are relative to the snake's heading: `0` forward, `1` turn right, `2` turn left.

**Observation** is 11 binary features:

- 3 danger flags: wall or body immediately straight, right, left
- 4 heading flags: left, right, up, down
- 4 food-location flags: food is left, right, above, below the head

**Rewards** (class constants on `SnakeEnv`): `-0.01` per step, `+10` for food,
`-10` for dying, and an extra `+10` for filling the board.

**Episode end**: `terminated` on wall/self collision or a full board, `truncated` after
`max_steps` steps. The reason is reported in `info["termination_reason"]`.
