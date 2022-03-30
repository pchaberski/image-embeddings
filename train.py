from utils.configuration import load_config
from utils.logger import configure_logger
import neptune.new as neptune
import os
from data.hm_data_module import HMDataModule


cfg = load_config('config.yml')
logger = configure_logger(__name__, cfg.get('logging_dir'), cfg.get('logging_level'))


if cfg.get('log_to_neptune'):
    logger.info('Initializing Neptune logging...')
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


settings_record = {
    'data_folder': os.path.basename(data_module.data_path),
    'image_size': data_module.image_size,
    'center': data_module.center,
    'center_params': str(data_module.center_params) if data_module.center else None,
    'batch_size': data_module.batch_size,
    'train_valid_ratio': data_module.train_valid_ratio
}
if cfg.get('log_to_neptune'):
    run['settings'] = settings_record

print(str(data_module.center_params))


logger.info('All done.')
