from utils.configuration import load_config
from utils.logger import configure_logger
import neptune.new as neptune
import os
from data.hm_data_module import HMDataModule


cfg = load_config('config.yml')
logger = configure_logger(__name__, cfg.get('logging_dir'), cfg.get('logging_level'))


if cfg.get('log_to_neptune'):
    run = neptune.init(
        project=os.path.join(cfg.get('neptune_username'), cfg.get('neptune_project')),
        api_token=cfg.get('neptune_api_token')
    )


data_module = HMDataModule(
        data_path=cfg.get('data_path'),
        batch_size=cfg.get('batch_size'),
        image_size=cfg.get('image_size'),
        center=cfg.get('center'),
        center_params=cfg.get('center_params')
)


print(data_module.train_valid_ratio)
