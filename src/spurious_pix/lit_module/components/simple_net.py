import torch
from torch import nn
import torch.nn.functional as tf

__all__ = ["SpPixCNN"]

class SpPixCNN(nn.Module):
    """
    Initialize the SimpleCNN for the spurious pixel task research experiment. 
    """
    def __init__(
        self,
        num_in: int = 1,
        num_classes: int = 10
    ) -> None:
        super().__init__()

        # [img_size] 28 -> [...] 24 -> [fm] 12 
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d( 
                in_channels=num_in,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        # -> [...] 10 -> [fm] 5
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
        )

        # in_features computed manually assuming stride 1 for convolutions, and 2 for pooling to be a valid computation and to reduce dimensionality, respectively        
        self.dense = nn.Sequential( 
            nn.Linear(in_features=7774, out_features=128), 
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
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