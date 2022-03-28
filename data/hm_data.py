
"""Dateset class for H&M images."""


import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class HMDataset(Dataset):
    """Class for H&M Images."""

    def __init__(
        self,
        data_path,
        image_size=[224, 224],
        normalize=False,
        normalization_params={'mean': None, 'std': None}
    ):
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.normalize = normalize
        self.normalization_params = normalization_params
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

        # Normalization
        if self.normalize:
            transf_list += [transforms.Normalize(
                mean=self.normalization_params['mean'],
                std=self.normalization_params['std']
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
    