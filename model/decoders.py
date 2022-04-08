"""Collection of decoder architectures."""


import torch.nn as nn
import torch.nn.functional as F


class DecoderBase(nn.Module):

    def __init__(
        self,
        image_size,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = image_size

        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(), 
            nn.Linear(128, self.image_size[0]*self.image_size[1]*3)
        )

    def forward(self, x):
        x = self.decoder(x)

        return x
