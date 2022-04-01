import pytorch_lightning as pl
import torch.nn.functional as F
from importlib import import_module


class LitHMAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        optimizer,
        optimizer_params,
        encoder,
        decoder
    ):
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        embedding = self.encoder(x)

        return embedding
    
    def training_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('valid_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

        return optimizer
        
