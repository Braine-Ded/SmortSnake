from src.environment import SnakeEnv
from src.pygame_view import run_manual


if __name__ == "__main__":
    env = SnakeEnv()
    run_manual(env)
