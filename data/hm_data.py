"""Dateset class for H&M images."""


import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd


class HMDataset(Dataset):
    """Class for H&M Images."""

    def __init__(
        self,
        data_path,
        center=False,
        center_params={'mean': None, 'std': None}
    ):
        super().__init__()
        self.data_path = data_path
        self.image_size = [128, 128]
        self.center = center
        self.center_params = center_params
        self.image_fnames = self._get_image_fnames(data_path)

    def _get_image_fnames(self, data_path):
        jpg_fnames = [file for file in os.listdir(data_path) if file.endswith('.jpg')]

        return jpg_fnames

    def _transform(self, image):
        transf_list = []

        # Resizing
        transf_list += [transforms.Resize(self.image_size)]

        # Convert to tensor
        transf_list += [transforms.ToTensor()]

        # Centering (optional)
        if self.center:
            transf_list += [transforms.Normalize(
                mean=self.center_params['mean'],
                std=self.center_params['std']
            )]

        transf = transforms.Compose(transf_list)

        return transf(image)

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = os.path.join(self.data_path, image_fname)
        image = Image.open(image_fpath).convert('RGB')

        image = self._transform(image)

        return image


def save_embeddings(
    embeddings: np.ndarray,
    output_dir: str,
    article_ids: list[str]
):
    colnames = ['f' + str(i + 1) for i in range(embeddings.shape[1])]
    df_embeddings = pd.DataFrame(embeddings, columns=colnames)
    df_embeddings.insert(0, 'article_id', article_ids)
    df_embeddings.to_csv(output_dir, index=False)
    