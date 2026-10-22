from src.config_loader import load_config
from src.pygame_view import run_manual

if __name__ == "__main__":
    config = load_config()
    env = config.make_env(config.SEED)
    run_manual(env)
