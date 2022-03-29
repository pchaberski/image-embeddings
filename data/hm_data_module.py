"""Pytorch Lightning module for H&M images."""


import pytorch_lightning as pl
from .hm_data import HMDataset
from torch.utils.data import DataLoader
import os


class HMDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        image_size=[224, 224],
        normalize=False,
        normalization_params={'mean': None, 'std': None}
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.normalize = normalize
        self.normalization_params = normalization_params
        self.train_valid_ratio = self._get_train_valid_ratio()

    def setup(self, stage='fit'):
        self.data_train = HMDataset(
            data_path=os.path.join(self.data_path, 'train'),
            image_size=self.image_size,
            normalize=self.normalize,
            normalization_params=self.normalization_params
        )

        self.data_valid = HMDataset(
            data_path=os.path.join(self.data_path, 'valid'),
            image_size=self.image_size,
            normalize=self.normalize,
            normalization_params=self.normalization_params
        )

    def train_dataloader(self):
        loader = DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data_valid, batch_size=self.batch_size)

        return loader

    def _get_train_valid_ratio(self):
        data_path_train = os.path.join(self.data_path, 'train')
        data_path_valid = os.path.join(self.data_path, 'valid')

        jpg_fnames_train = [file for file in os.listdir(data_path_train) if file.endswith('.jpg')]
        jpg_fnames_valid = [file for file in os.listdir(data_path_valid) if file.endswith('.jpg')]

        return len(jpg_fnames_train)/(len(jpg_fnames_train) + len(jpg_fnames_valid))
