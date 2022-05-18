"""Collection of decoder architectures."""


import torch.nn as nn
import torch.nn.functional as F
import torch


class DecoderLinearBase(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(), 
            nn.Linear(128, self.image_size[0]*self.image_size[1]*3)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x


class DecoderLinearLin1024(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_size, 1024),
            nn.ReLU(), 
            nn.Linear(1024, self.image_size[0]*self.image_size[1]*3)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x


class DecoderConvBase(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 262144),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(
            dim=1, 
            unflattened_size=(256, 32, 32)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 16, 2, stride = 2, padding = 0),
            nn.ConvTranspose2d(16, 3, 2, stride = 2)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x


class DecoderConvLin1024(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(self.embedding_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 262144),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(
            dim=1, 
            unflattened_size=(256, 32, 32)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 16, 2, stride = 2, padding = 0),
            nn.ConvTranspose2d(16, 3, 2, stride = 2)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x


class DecoderConvCompr(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32768),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(
            dim=1, 
            unflattened_size=(32, 32, 32)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 2, stride = 2, padding = 0),
            nn.ConvTranspose2d(64, 3, 2, stride = 2)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x


class DecoderConvCompr3Layer(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 8192),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(
            dim=1, 
            unflattened_size=(32, 16, 16)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 2, stride = 2, padding = 0),
            nn.ConvTranspose2d(64, 128, 2, stride = 2),
            nn.ConvTranspose2d(128, 3, 2, stride = 2)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x
    

class DecoderConvCompr3LayerV2(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(self.embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 8192),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(
            dim=1, 
            unflattened_size=(32, 16, 16)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 2, stride = 2, padding = 0),
            nn.ConvTranspose2d(64, 128, 2, stride = 2),
            nn.ConvTranspose2d(128, 3, 2, stride = 2)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x
    

class DecoderConvCompr3LayerV3(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(self.embedding_size, 8192),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(
            dim=1, 
            unflattened_size=(32, 16, 16)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 2, stride = 2, padding = 0),
            nn.ConvTranspose2d(64, 128, 2, stride = 2),
            nn.ConvTranspose2d(128, 3, 2, stride = 2)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x


class DecoderConvLarger1stKernel(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(self.embedding_size, 7200),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(
            dim=1, 
            unflattened_size=(32, 15, 15)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 4, stride = 2),
            nn.ConvTranspose2d(64, 128, 2, stride = 2),
            nn.ConvTranspose2d(128, 3, 2, stride = 2)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x


class DecoderConvLarger2Kernels(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(self.embedding_size, 7200),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(
            dim=1, 
            unflattened_size=(32, 15, 15)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 4, stride = 2),
            nn.ConvTranspose2d(64, 128, 2, stride = 2),
            nn.ConvTranspose2d(128, 3, 2, stride = 2)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x


class DecoderConvLarger2KernelsLeaky(nn.Module):

    def __init__(
        self,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(self.embedding_size, 7200),
            nn.LeakyReLU()
        )

        self.unflatten = nn.Unflatten(
            dim=1, 
            unflattened_size=(32, 15, 15)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 4, stride = 2),
            nn.ConvTranspose2d(64, 128, 2, stride = 2),
            nn.ConvTranspose2d(128, 3, 2, stride = 2)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x

