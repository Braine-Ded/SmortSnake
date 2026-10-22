import importlib

# The contract every module in configs/ fulfils (see config.example.py).
REQUIRED = ("NAME", "SEED", "MAX_EPISODES", "make_env", "make_agent")


def load_config():
    """Import the git-ignored config.py and check it selects a full setup."""
    try:
        config = importlib.import_module("config")
    except ModuleNotFoundError as error:
        if error.name != "config":
            raise
        raise SystemExit(
            "No config.py found. Create one with: cp config.example.py config.py"
        ) from None

    missing = [name for name in REQUIRED if not hasattr(config, name)]
    if missing:
        raise SystemExit(
            f"config.py is missing {', '.join(missing)}. It should select a setup, "
            "e.g. `from configs.dqn_features import *` (see config.example.py)."
        )
    return config
