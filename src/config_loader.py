import importlib
import os

# The contract every module in configs/ fulfils (see config.example.py).
# ALGO ... ENV are what the run name is built from (see src/naming.py).
REQUIRED = (
    "NAME", "SEED", "MAX_EPISODES", "make_env", "make_agent",
    "ALGO", "POLICY", "NET", "OBS", "ARCH", "GAMMA", "ENV",
)

# Set to a module path (e.g. configs.dqn_cnn_ego) to bypass config.py, so
# several runs with different configs can train side by side.
CONFIG_ENV_VAR = "RLSNAKE_CONFIG"


def load_config():
    """Import the git-ignored config.py and check it selects a full setup."""
    module_name = os.environ.get(CONFIG_ENV_VAR) or "config"
    try:
        config = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        if module_name != "config":
            raise SystemExit(f"{CONFIG_ENV_VAR}={module_name} does not exist.") from None
        raise SystemExit(
            "No config.py found. Create one with: cp config.example.py config.py"
        ) from None

    missing = [name for name in REQUIRED if not hasattr(config, name)]
    if missing:
        raise SystemExit(
            f"{module_name} is missing {', '.join(missing)}. It should select a setup, "
            "e.g. `from configs.dqn_features import *` (see config.example.py)."
        )
    return config
