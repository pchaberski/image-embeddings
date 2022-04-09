import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from data.hm_data import HMDataset
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from neptune.new.types import File


class LitHMAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        optimizer,
        optimizer_params,
        encoder,
        decoder,
        data_path: str,
        batch_size: int,
        num_workers = 1,
        center=False,
        center_params={'mean': None, 'std': None},
        run = None
    ):
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.encoder = encoder
        self.decoder = decoder
        self.run = run
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.center = center
        self.center_params = center_params
        self.train_valid_ratio = self._get_train_valid_ratio()
        self.image_size = [224, 224]

    def forward(self, x):
        embedding = self.encoder(x)

        return embedding
    
    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat.flatten(), x.flatten())
        if self.run:
            self.run['metrics/batch/train_loss'].log(loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat.flatten(), x.flatten())
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

            data_iter = iter(self.predict_dataloader())
            img_sample = data_iter.next()[:16, :, :, :]

            img_sample_original = torchvision.utils.make_grid(img_sample, 4, 4)
            img_sample_original_resized = self._resize_tensor(img_sample_original)

            self.encoder.train(False)
            self.decoder.train(False)

            embeddings = self.encoder(img_sample.reshape(16, 3*self.image_size[0]*self.image_size[1]))

            img_sample_decoded = self.decoder(embeddings).reshape(16, 3, self.image_size[0], self.image_size[1])
            img_sample_reconstructed = torchvision.utils.make_grid(img_sample_decoded, 4, 4)
            img_sample_reconstructed_resized = self._resize_tensor(img_sample_reconstructed)

            img_sample_original_resized_np_t = np.transpose(img_sample_original_resized.numpy(), (1, 2, 0))
            img_sample_reconstructed_resized_np_t = np.transpose(img_sample_reconstructed_resized.numpy(), (1, 2, 0))

            self.run[f'results/img_sample/original'].upload(File.as_image(img_sample_original_resized_np_t))
            self.run[f'results/img_sample/reconstructed'].log(File.as_image(img_sample_reconstructed_resized_np_t))
            self.run['results/example_embeddings'].upload(File.as_html(pd.DataFrame(embeddings.numpy())))

            self.encoder.train(True)
            self.decoder.train(True)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

        return optimizer

    def setup(self, stage=None):
        self.data_train = HMDataset(
            data_path=os.path.join(self.data_path, 'train'),
            center=self.center,
            center_params=self.center_params
        )

        self.data_valid = HMDataset(
            data_path=os.path.join(self.data_path, 'valid'),
            center=self.center,
            center_params=self.center_params
        )

        self.data_predict = HMDataset(
            data_path=os.path.join(self.data_path, 'valid'),
            center=self.center,
            center_params=self.center_params
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.data_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )

        return loader

    def predict_dataloader(self):
        loader = DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )

        return loader

    def _get_train_valid_ratio(self):
        data_path_train = os.path.join(self.data_path, 'train')
        data_path_valid = os.path.join(self.data_path, 'valid')

        jpg_fnames_train = [file for file in os.listdir(data_path_train) if file.endswith('.jpg')]
        jpg_fnames_valid = [file for file in os.listdir(data_path_valid) if file.endswith('.jpg')]

        return len(jpg_fnames_train)/(len(jpg_fnames_train) + len(jpg_fnames_valid))

    def _resize_tensor(self, tensor, height: int = 400, width: int = 400):
        transform = torchvision.transforms.Resize((height, width))
        resized_tensor = transform(tensor)

        return resized_tensor
