import sys

sys.path.append('..')

from utils.configuration import load_config
from data.hm_data import HMDataset
import torch
import numpy as np
import os
from tqdm import tqdm

cfg = load_config('../config.yml')

data_train = HMDataset(
    data_path=os.path.join(cfg.get('data_path'), 'train'),
    image_size=cfg.get('image_size'),
    center=cfg.get('center'),
    center_params=cfg.get('center_params')
)

means = {
    'r': [],
    'g': [],
    'b': []
}
stds = {
    'r': [],
    'g': [],
    'b': []
}

print(f'Images path: {data_train.data_path}')
print('Processing images to extract means and standard deviations...')
for idx in tqdm(range(len(data_train))):
    img = data_train[idx].detach().cpu()
    for i, k in enumerate(means):
        means[k].append(torch.mean(img[i, :, :]))
        stds[k].append(torch.std(img[i, :, :]))

print('Results:')
print(f'  mean: {[round(float(torch.mean(torch.stack(v)).numpy()), 4) for v in means.values()]}')
print(f'  std: {[round(float(torch.mean(torch.stack(v)).numpy()), 4) for v in stds.values()]}')


