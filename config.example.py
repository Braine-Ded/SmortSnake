# Picks which setup train.py, eval.py and play.py run. Copy to config.py
# (git-ignored) and change the import to switch setups:
#
#     cp config.example.py config.py
#
# Each module in configs/ is a complete setup: hyperparameters plus
# make_env(seed) and make_agent(env, device, seed), which build the
# observation, network, optimizer, loss, buffer, policy and agent.
# To try a new architecture, copy a config file, edit it, and import it here.
# Don't override values below the import: make_env/make_agent read the
# config module's own globals, so the override would be silently ignored.

from configs.dqn_features import *
