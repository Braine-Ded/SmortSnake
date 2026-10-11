| File | Responsibility |
| --- | --- |
| `environment.py` | Snake state, movement, food spawning, collisions, rewards, episode termination |
| | |
| `pygame_view.py` | Display the environment, render snake and food, handle keyboard input |
| | |
| `model.py` | Neural network used to estimate action values |
| | |
| `agent.py` | Action selection, exploration, learning |
| | |
| `replay_buffer.py` | Store transitions for replay-based learning |
| | |
| `train.py` | Run training episodes, update the agent, record metrics |
| | |
| `config.py` | Board dimensions, learning rate, discount factor, reward constants, training settings |
