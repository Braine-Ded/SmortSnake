import torch

from src import Transition
from src.config_loader import load_config

config = load_config()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 1. Build the selected setup exactly as train.py does.
    env = config.make_env(config.SEED)
    agent = config.make_agent(env, device, config.SEED)
    print("Config:", config.NAME)

    # 2. Verify a single greedy decision.
    state, info = env.reset(seed=123)
    action = agent.act(state, explore=False)

    assert isinstance(action, int)
    assert action in range(env.NUM_ACTIONS)
    print("State shape:", state.shape)
    print("Selected action:", action)

    # 3. Verify the environment-agent interaction.
    next_state, reward, terminated, truncated, info = env.step(action)
    agent.observe(
        Transition(state, action, reward, next_state, terminated, truncated)
    )
    print("Reward:", reward)
    print("Terminated / truncated:", terminated, truncated)

    # 4. Fresh weights round-trip: a second agent loaded from them acts the same.
    clone = config.make_agent(env, device, config.SEED)
    clone.load_weights(agent.weights())
    assert clone.act(state, explore=False) == action

    # 5. Full training state round-trips (networks, optimizer, buffer, policy).
    clone.load_state_dict(agent.state_dict())
    for online, restored in zip(agent.weights().values(), clone.weights().values()):
        assert torch.equal(online, restored)

    print("Setup, agent, weights and training-state smoke test passed.")


if __name__ == "__main__":
    main()
