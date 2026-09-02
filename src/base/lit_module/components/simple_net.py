import torch
from torch import nn
import torch.nn.functional as tf

__all__ = ["SimpleCNN"]

class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_in: int = 1,
        num_classes: int = 10
    ) -> None:
        """Initialize a hardcoded SimpleCNN for the spurious pixel task
        """
        super().__init__()

        # [img_size] 28 -> [...] 24 -> [fm] 12 
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d( 
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # -> [...] 10 -> [fm] 5
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 5*5*16
        self.dense = nn.Sequential( 
            nn.Linear(in_features=144, out_features=30),
            nn.ReLU(),
            nn.Linear(in_features=30, out_features=10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """

        x = tf.relu(self.conv_layer1(x)) 
        x = tf.relu(self.conv_layer2(x))
        x = torch.flatten(x, 1)
        x = self.dense(x)

        return x