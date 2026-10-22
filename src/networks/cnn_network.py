import torch.nn as nn

from .base import QNetwork

class CNN(nn.Module, QNetwork):
    """Q-network for (C, H, W) board planes, e.g. PlaneObservation.

    No pooling: on a board this small every cell's exact position matters.
    """

    def __init__(
        self,
        input_shape,
        conv1_channels=32,
        conv2_channels=64,
        hidden_size=256,
        output_size=3,
    ):
        super().__init__()
        channels, height, width = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(channels, conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(conv2_channels * height * width, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # Raw Q-values, no output activation (same as the MLP).
        return self.head(self.features(x))
