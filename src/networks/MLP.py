import torch
import torch.nn as nn
import torch.nn.functional as F

from base import QNetwork

class MLP(nn.Module, QNetwork):

    def __init__(
        self,
        input_size=11,
        hidden1_size=64,
        hidden2_size=32,
        output_size=3
    ):
        super().__init__()

        self.input_layer = nn.Linear(input_size, hidden1_size)
        self.hidden_layer = nn.Linear(hidden1_size, hidden2_size)
        self.output_layer = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))

        # No normalization using Sigmoid or Softmaz used here
        # Since its better for the model to recieve actual Q values
        return self.output_layer(x)
