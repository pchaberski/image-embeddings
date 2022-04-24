"""Collection of encoder architectures."""


import torch.nn as nn
import torch.nn.functional as F


class EncoderLinearBase(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
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


class EncoderLinearLin1024(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.encoder = nn.Sequential(
            nn.Linear(self.image_size[0]*self.image_size[1]*3, 1024), 
            nn.ReLU(), 
            nn.Linear(1024, self.embedding_size)
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
        self.image_size = [128, 128]
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
            nn.Linear(262144, 128),
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


class EncoderConvLin1024(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):

        super().__init__()
        self.image_size = [128, 128]
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
            nn.Linear(262144, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        return x


class EncoderConvCompr(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):

        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(32768, 128),
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


class EncoderConvCompr3Layer(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):

        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(8192, 128),
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
    

class EncoderConvCompr3LayerV2(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):

        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        return x
    

class EncoderConvCompr3LayerV3(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):

        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(8192, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        return x