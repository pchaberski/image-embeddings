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
import torch
from tqdm import tqdm


class LitHMAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        batch_size: int,
        encoder,
        decoder,
        num_workers = 0,
        optimizer = None,
        optimizer_params = None,
        data_path: str = None,
        center=False,
        center_params={'mean': None, 'std': None},
        run = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.run = run
        self.data_path = data_path
        self.num_workers = num_workers
        self.center = center
        self.center_params = center_params
        self.train_valid_ratio = self._get_train_valid_ratio()
        self.image_size = [128, 128]

    def forward(self, x):
        embedding = self.encoder(x)

        return embedding
    
    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat.flatten(), x.flatten())
        self.log('loss', loss)
        if self.run:
            self.run['metrics/batch/train_loss'].log(loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat.flatten(), x.flatten())
        self.log('val_loss', loss)
        if self.run:
            self.run['metrics/batch/valid_loss'].log(loss)

        return {'val_loss': loss}

    def training_epoch_end(self, outputs):
        batch_losses = np.array([])
        for results_dict in outputs:
            batch_losses = np.append(batch_losses, results_dict['loss'].cpu())
        epoch_loss = batch_losses.mean()
        if self.run:
            self.run['metrics/epoch/train_loss'].log(epoch_loss)

    def validation_epoch_end(self, outputs):
        batch_losses = np.array([])
        for results_dict in outputs:
            batch_losses = np.append(batch_losses, results_dict['val_loss'].cpu())
        epoch_loss = batch_losses.mean()
        if self.run:
            self.run['metrics/epoch/valid_loss'].log(epoch_loss)

            data_iter = iter(self.test_dataloader())
            img_sample = data_iter.next()[:16, :, :, :]

            if torch.cuda.memory_allocated(0) > 0:
                img_sample = img_sample.cuda()

            img_sample_original = torchvision.utils.make_grid(img_sample, 4, 4)
            img_sample_original_resized = self._resize_tensor(img_sample_original)

            self.encoder.train(False)
            self.decoder.train(False)

            embeddings = self.encoder(img_sample.reshape(-1, 3*self.image_size[0]*self.image_size[1]))

            img_sample_decoded = self.decoder(embeddings).reshape(-1, 3, self.image_size[0], self.image_size[1])
            img_sample_reconstructed = torchvision.utils.make_grid(img_sample_decoded, 4, 4)
            img_sample_reconstructed_resized = self._resize_tensor(img_sample_reconstructed)

            if torch.cuda.memory_allocated(0) > 0:
                img_sample_original_resized = img_sample_original_resized.cpu()
                img_sample_reconstructed_resized = img_sample_reconstructed_resized.cpu()
                embeddings = embeddings.cpu()

            img_sample_original_resized_np_t = np.transpose(img_sample_original_resized.numpy(), (1, 2, 0))
            img_sample_reconstructed_resized_np_t = np.transpose(img_sample_reconstructed_resized.numpy(), (1, 2, 0))

            self.run[f'results/img_sample/original'].upload(File.as_image(img_sample_original_resized_np_t))
            self.run[f'results/img_sample/reconstructed'].log(File.as_image(img_sample_reconstructed_resized_np_t))
            self.run['results/example_embeddings'].upload(File.as_html(pd.DataFrame(embeddings.numpy())))

            self.encoder.train(True)
            self.decoder.train(True)

    def on_train_epoch_start(self):
        if self.run:
            self.run['curr_epoch'] = self.current_epoch

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

        return optimizer

    def setup(self, stage=None):
        self.data_train = HMDataset(
            data_path=os.path.join(self.data_path, 'train') if self.data_path is not None else None,
            center=self.center,
            center_params=self.center_params
        )
        self.data_valid = HMDataset(
            data_path=os.path.join(self.data_path, 'valid') if self.data_path is not None else None,
            center=self.center,
            center_params=self.center_params
        )

        self.data_test = HMDataset(
            data_path=os.path.join(self.data_path, 'valid') if self.data_path is not None else None, # only to print examples of reconstructions
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

    def test_dataloader(self):
        loader = DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )

        return loader


    def calculate_embeddings(self, data_path):
        self.encoder.train(False)

        embeddings = np.empty((0, self.encoder.embedding_size))

        data_predict = HMDataset(
            data_path=data_path,
            center=self.center,
            center_params=self.center_params
        )

        loader_predict = DataLoader(
            data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False
        )

        for batch in tqdm(iter(loader_predict)):
            embeddings_batch = self.encoder(batch.reshape(-1, 3*self.image_size[0]*self.image_size[1]))
            embeddings_batch = embeddings_batch.detach().numpy()
            embeddings = np.concatenate((embeddings, embeddings_batch), axis=0)

        return embeddings

    def get_num_params(self):
        total_params_encoder = sum(p.numel() for p in self.encoder.parameters())
        trainable_params_encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_params_decoder = sum(p.numel() for p in self.decoder.parameters())
        trainable_params_decoder = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

        num_params = {
            'total': total_params_encoder + total_params_decoder,
            'total_trainable': trainable_params_encoder + trainable_params_decoder,
            'encoder': total_params_encoder,
            'encoder_trainable': trainable_params_encoder,
            'decoder': total_params_decoder,
            'decoder_trainable': trainable_params_decoder 
        }

        return num_params

    def _get_train_valid_ratio(self):
        if self.data_path is not None:
            data_path_train = os.path.join(self.data_path, 'train')
            data_path_valid = os.path.join(self.data_path, 'valid')

            jpg_fnames_train = [file for file in os.listdir(data_path_train) if file.endswith('.jpg')]
            jpg_fnames_valid = [file for file in os.listdir(data_path_valid) if file.endswith('.jpg')]

            return len(jpg_fnames_train)/(len(jpg_fnames_train) + len(jpg_fnames_valid))
        else:
            return None

    def _resize_tensor(self, tensor, width: int = 400):
        h_w_ratio = tensor.shape[1] / tensor.shape[2]
        height = int(h_w_ratio*width)

        transform = torchvision.transforms.Resize((height, width))
        resized_tensor = transform(tensor)

        return resized_tensor
