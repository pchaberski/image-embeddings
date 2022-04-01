from utils.configuration import load_config
from utils.logger import configure_logger
import neptune.new as neptune
import os
from data.hm_data_module import HMDataModule
from model.autoencoder_module import LitHMAutoEncoder
import pytorch_lightning as pl
from importlib import import_module


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
    num_workers=cfg.get('num_workers'),
    image_size=cfg.get('image_size'),
    center=cfg.get('center'),
    center_params=cfg.get('center_params')
)


model = LitHMAutoEncoder(
    optimizer=getattr(import_module('torch.optim'), cfg.get('optimizer')),
    optimizer_params=cfg.get('optimizer_params'),
    encoder=getattr(import_module('model.encoders'), cfg.get('encoder'))(cfg.get('image_size')),
    decoder=getattr(import_module('model.decoders'), cfg.get('decoder'))(cfg.get('image_size'))
)


trainer = pl.Trainer(
    max_epochs=cfg.get('num_epochs'),
    gpus=cfg.get('num_gpus')
)


settings_record = {
    'data_folder': os.path.basename(data_module.data_path),
    'image_size': data_module.image_size,
    'center': data_module.center,
    'center_params': str(data_module.center_params) if data_module.center else None,
    'batch_size': data_module.batch_size,
    'train_valid_ratio': data_module.train_valid_ratio,
    'num_epochs': trainer.max_epochs,
    'num_workers': data_module.num_workers,
    'num_gpus': trainer.gpus,
    'encoder': cfg.get('encoder'),
    'decoder': cfg.get('decoder')
}
if cfg.get('log_to_neptune'):
    run['settings'] = settings_record


trainer.fit(model, data_module)


logger.info('All done.')
