import os
import shutil
from tqdm import tqdm


INPUT_DIR = '/Users/pchaberski/data/kaggle_hm/original'
OUTPUT_DIR = '/Users/pchaberski/data/kaggle_hm/single_folder'


def gather_to_one_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for path, _, fnames in os.walk(input_dir):
        print(f'Copying from path: {path}')
        for fname in tqdm(fnames):
            src = os.path.join(path, fname)
            dest = os.path.join(output_dir, fname)
            shutil.copy2(src, dest)


if __name__ == "__main__":
    gather_to_one_folder(INPUT_DIR, OUTPUT_DIR)
