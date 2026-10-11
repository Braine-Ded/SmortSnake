# RLSnake

A Snake environment with a pygame viewer, set up for training a DQN agent.
The environment, viewer, replay buffer, Q-network and greedy policy are in place;
the DQN agent, epsilon-greedy policy and training/testing scripts are still to be written.

## Usage

Play the game manually (Arrows/WASD to move, R to restart, Esc/Q to quit):

```
python play.py
```

Watch a trained policy (any callable that maps an observation to an action 0/1/2):

```python
from src.pygame_view import run_simulated
from src.policies.GreedyPolicy import GreedyPolicy

policy = GreedyPolicy(model, device="cpu")
run_simulated(env, policy.select_action, episodes=5)
```

Requires `numpy`, `pygame` and `torch`. Run everything from the project root.

## Setup

`config.py` is git-ignored so per-run hyperparameters don't clutter history.
On a fresh clone, create it from the template:

```
cp config.example.py config.py
```

Training writes `checkpoints/` and `logs/`, which are also git-ignored.
Training can be stopped with Ctrl+C and continued later:

```
python train.py --resume --log <same-log-name>
```

## Project layout

| File | Responsibility | Status |
| --- | --- | --- |
| `config.example.py` | Template for `config.py` (git-ignored): board size and training hyperparameters (learning rate, gamma, epsilon schedule, batch size, replay capacity, target update frequency, max episodes) | Done |
| `play.py` | Entry point for manual play | Done |
| `train.py` | Run training episodes, update the agent, record metrics | Empty |
| `test.py` | Evaluate a trained agent | Empty |
| `src/environment.py` | Snake state, movement, food spawning, collisions, rewards, episode termination | Done |
| `src/pygame_view.py` | Render the environment, manual play (`run_manual`), policy playback with CSV metrics (`run_simulated`) | Done |
| `src/metadata.py` | Shared types: `Collision` enum, `Transition` and `TransitionBatch` dataclasses | Done |
| `src/agents/base.py` | Abstract `Agent` interface (`act`, `observe`, `learn`) | Done |
| `src/agents/DQNAgent.py` | DQN agent: action selection, exploration, learning | Stub |
| `src/buffers/base.py` | Abstract `Buffer` interface | Done |
| `src/buffers/ReplayBuffer.py` | Fixed-capacity, seeded replay memory with uniform sampling | Done |
| `src/networks/base.py` | Abstract `QNetwork` interface | Done |
| `src/networks/MLP.py` | 11 -> 64 -> 32 -> 3 MLP that outputs raw Q-values | Done |
| `src/policies/base.py` | Abstract `Policy` interface | Done |
| `src/policies/GreedyPolicy.py` | Picks the action with the highest Q-value | Done |
| `src/policies/EpsilonGreedyPolicy.py` | Random action with probability epsilon, otherwise greedy | Empty |

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
