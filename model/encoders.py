"""Collection of encoder architectures."""


import torch.nn as nn
import torch.nn.functional as F


class EncoderBase(nn.Module):

    def __init__(
        self,
        image_size
    ):
        super().__init__()
        self.image_size = image_size

        self.encoder = nn.Sequential(
            nn.Linear(self.image_size[0]*self.image_size[1]*3, 128), 
            nn.ReLU(), 
            nn.Linear(128, 32)
        )

    def forward(self, x):
        x = self.encoder(x)

        return x