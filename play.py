from src.environment import SnakeEnv
from src.pygame_view import run_manual

import config

if __name__ == "__main__":
    env = SnakeEnv(config.GRID_SIZE, 3, tuple(i//2 for i in config.GRID_SIZE))
    run_manual(env)
