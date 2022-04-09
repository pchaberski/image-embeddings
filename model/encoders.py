"""Collection of encoder architectures."""


import torch.nn as nn
import torch.nn.functional as F


class EncoderBase(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [224, 224]
        self.embedding_size = embedding_size

        self.encoder = nn.Sequential(
            nn.Linear(self.image_size[0]*self.image_size[1]*3, 128), 
            nn.ReLU(), 
            nn.Linear(128, self.embedding_size)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)

        return x


class EncoderConvBase(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):

        super().__init__()
        self.image_size = [224, 224]
        self.embedding_size = embedding_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(802816, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        return x