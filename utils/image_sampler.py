import os
from datetime import datetime
import random
from tqdm import tqdm
import shutil


INPUT_DIR = '/Users/pchaberski/data/kaggle_hm/single_folder'  # all images should be in a single folder
OUTPUT_DIR = '/Users/pchaberski/data/kaggle_hm/samples'
SAMPLE_USING_RATE = False  # Whether to use sample rate (0.0-1.0, if True) or absolute value of wanted samples (if False)
SAMPLE_RATE = 0.01
NUM_SAMPLES = 1000
MAKE_VALID = True
TRAIN_VALID_RATIO = 0.9
SEED = 99


def sample_images(
    input_dir, output_dir,
    sample_using_rate: bool = True,
    sample_rate: float = 0.01,
    num_samples: int = None,
    make_valid: bool = False,
    train_valid_ratio: float = 0.9,
    seed: int = 99
):
    file_list = os.listdir(input_dir)
    num_images = len(file_list)

    if sample_using_rate:
        assert sample_rate > 0. and sample_rate <= 1., 'Sample rate should be > 0 and <=1'
        num_samples = int(sample_rate*num_images)
    else:
        assert isinstance(num_samples, int), 'Number of samples should be of type `int`'
        assert num_samples > 0 and num_samples <= num_images, 'Number of samples should be >0 and <= number of files'

    sampling_ts = datetime.now().strftime('%Y%m%d%H%M%S')

    random.seed(seed)
    sample_images = random.sample(file_list, num_samples)

    if make_valid:
        assert train_valid_ratio > 0. and train_valid_ratio <= 1., 'Train/valid ratio should be > 0 and <=1'
        num_train_samples = int(train_valid_ratio*len(sample_images))
        train_sample = random.sample(sample_images, num_train_samples)
        valid_sample = list(set(sample_images) - set(train_sample))   
    else:
        train_sample = sample_images

    output_subdir = os.path.join(output_dir, f'smpl_{num_samples}_{sampling_ts}')
    output_subdir_train = os.path.join(output_subdir, 'train')
    os.makedirs(output_subdir_train)
    print('Copying training images...')
    for fname in tqdm(train_sample):
        src = os.path.join(input_dir, fname)
        dest = os.path.join(output_subdir_train, fname)
        shutil.copy2(src, dest)
    
    if make_valid:
        output_subdir_valid = os.path.join(output_subdir, 'valid')
        os.makedirs(output_subdir_valid)
        print('Copying validation images...')
        for fname in tqdm(valid_sample):
            src = os.path.join(input_dir, fname)
            dest = os.path.join(output_subdir_valid, fname)
            shutil.copy2(src, dest)


if __name__ == "__main__":
    sample_images(INPUT_DIR, OUTPUT_DIR, SAMPLE_USING_RATE, SAMPLE_RATE, NUM_SAMPLES, SEED)