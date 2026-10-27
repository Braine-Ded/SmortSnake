# Run naming

Every run is named from its config by `run_name()` in `src/naming.py`. Logs,
checkpoints and plots all use that name, so a file name says exactly what was
trained.

```
{algo}_{policy}_{net}_{obs}_{gamma}_{episodes}_{arch}[_{deviation}...]
```

Examples:

```
dqn_egp_mlp_feat_9_5k_064032                 DQN, epsilon-greedy, MLP 64/32 on the 11 features, gamma 0.9, 5000 episodes
dqn_egp_cnn_ego_95_10k_016032064_sp010       DQN, epsilon-greedy, CNN 16/32/64 on the egocentric board, gamma 0.95, 10000 episodes, step penalty 0.10
dqn_smp_mlp_feat_9_5k_064032                 the same feature setup with softmax exploration
```

## Fields

Fields are separated by `_`, so no label may contain one.

| Field | Set in the config as | Values |
| --- | --- | --- |
| algo | `ALGO` | `dqn` |
| policy | `POLICY` | `egp` (epsilon-greedy), `smp` (softmax / Boltzmann) |
| net | `NET` | `mlp`, `cnn` |
| obs | `OBS` | `feat` (11 features), `plane` (board planes, fixed directions), `ego` (board planes rotated so the head faces up) |
| gamma | `GAMMA` | only the digits after the decimal point: 0.9 -> `9`, 0.85 -> `85`, 0.95 -> `95` |
| episodes | `MAX_EPISODES` | in thousands: 10000 -> `10k` (counts that aren't whole thousands are written out, e.g. `2500`) |
| arch | `ARCH` | 3 digits per layer: CNN conv1, conv2, hidden (16/32/64 -> `016032064`); MLP hidden1, hidden2 (64/32 -> `064032`) |

New algorithms and policies get a short lowercase label of their own (for
example `ppo` and `cat` for a categorical policy-gradient policy).

## Deviations

Only `ENV` fields that differ from the `EnvSettings()` defaults are added, in
this order:

| Token | Meaning | Default |
| --- | --- | --- |
| `gs8x8` | grid size | 10x10 |
| `ss4` | start size | 3 |
| `ms500`, `msnone` | max steps per episode (`none` = no limit) | 1000 |
| `sp010` | step penalty, in hundredths (0.10) | 0.01 |
| `sr005` | step *reward* when positive, in hundredths (0.05) | |
| `fr20` | food reward | 10 |
| `dr5` | death penalty | 10 |
| `wb50` | win bonus | 10 |

Non-integer values replace the dot with `p`: a step penalty of 0.015 is
`sp1p5`, a win bonus of 12.5 is `wb12p5`.

## Files

| File | Location |
| --- | --- |
| `train_<run>.csv` | `logs/` |
| `train_<run>.json` (every setting, including those not in the name) | `logs/` |
| `eval_<run>.csv` | `logs/` |
| `{best,final,state}_<run>.pt` | `checkpoints/<config NAME>/` |
| `train_<run>.png`, `eval_<run>.png` (dashboards; single panels in a `<log name>/` folder) | `analysis/plots/` |

## Not in the name

Hyperparameters such as the learning rate, the epsilon or temperature schedule,
the replay size and **the seed** are not part of the name. They are recorded in
`train_<run>.json`. Two configs that differ only in these (for example the same
setup with `SEED = 68`) get the same run name, so pass `--log` with a distinct
name and rename the checkpoints, or they will overwrite each other.
