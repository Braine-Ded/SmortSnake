# Default configuration. Copy to config.py (git-ignored) and edit per run:
#
#     cp config.example.py config.py

# ---------- Environment ----------
GRID_SIZE = (10, 10)

# ---------- Optimisation ----------
LEARNING_RATE = 1e-3
GAMMA = 0.9

# ---------- Exploration (epsilon decays once per episode) ----------
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# ---------- DQN ----------
BATCH_SIZE = 64
REPLAY_CAPACITY = 100_000      # keep this bounded; the buffer is preallocated
TARGET_UPDATE_FREQ = 200       # in gradient updates

# ---------- Run length ----------
MAX_EPISODES = 5000
