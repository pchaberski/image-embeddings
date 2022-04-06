import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np


class LitHMAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        optimizer,
        optimizer_params,
        encoder,
        decoder,
        run=None
    ):
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.encoder = encoder
        self.decoder = decoder
        self.run = run

    def forward(self, x):
        embedding = self.encoder(x)

        return embedding
    
    def training_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        if self.run:
            self.run['metrics/batch/train_loss'].log(loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        if self.run:
            self.run['metrics/batch/valid_loss'].log(loss)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        batch_losses = np.array([])
        for results_dict in outputs:
            batch_losses = np.append(batch_losses, results_dict["loss"])
        epoch_loss = batch_losses.mean()
        if self.run:
            self.run['metrics/epoch/train_loss'].log(epoch_loss)

    def validation_epoch_end(self, outputs):
        batch_losses = np.array([])
        for results_dict in outputs:
            batch_losses = np.append(batch_losses, results_dict["loss"])
        epoch_loss = batch_losses.mean()
        if self.run:
            self.run['metrics/epoch/valid_loss'].log(epoch_loss)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

        return optimizer
        
