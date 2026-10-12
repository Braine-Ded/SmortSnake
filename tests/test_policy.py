import torch

import config
from src import SnakeEnv, Transition
from src.buffers import ReplayBuffer
from src.networks import MLP
from src.policies import GreedyPolicy


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = SnakeEnv(
        config.GRID_SIZE,
        3,
        tuple(size // 2 for size in config.GRID_SIZE),
        seed=config.SEED,
    )

    # 1. Create the online network.
    model = MLP(
        input_size=env.OBSERVATION_SIZE,
        output_size=env.NUM_ACTIONS,
    ).to(device)

    # 2. Create a matching, initially identical target network.
    target_model = MLP(
        input_size=env.OBSERVATION_SIZE,
        output_size=env.NUM_ACTIONS,
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    target_model.requires_grad_(False)

    # 3. Set up the optimizer and replay buffer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )
    buffer = ReplayBuffer(
        capacity=config.REPLAY_CAPACITY,
        seed=config.SEED,
    )

    # 4. Inject the online model into the greedy policy.
    policy = GreedyPolicy(model, device=device)

    # 5. Verify a single decision.
    state, info = env.reset(seed=123)
    action = policy.select_action(state)

    assert isinstance(action, int)
    assert action in range(env.NUM_ACTIONS)
    print("State shape:", state.shape)
    print("Selected action:", action)

    # 6. Verify the environment-policy interaction.
    next_state, reward, terminated, truncated, info = env.step(action)

    buffer.push(
        Transition(state, action, reward, next_state, terminated, truncated)
    )

    assert len(buffer) == 1
    print("Buffer length:", len(buffer))
    print("Reward:", reward)
    print("Terminated / truncated:", terminated, truncated)

    # 7. Verify target and online parameters begin identical.
    for online, target in zip(model.parameters(), target_model.parameters()):
        assert torch.equal(online, target)

    print("Online network, target network, policy and buffer smoke test passed.")


if __name__ == "__main__":
    main()
