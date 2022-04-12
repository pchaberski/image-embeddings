from utils.configuration import load_config
from utils.logger import configure_logger
import neptune.new as neptune
import os
from model.autoencoder_module import LitHMAutoEncoder
import pytorch_lightning as pl
from importlib import import_module
from datetime import datetime


cfg = load_config('config.yml')
logger = configure_logger(__name__, cfg.get('logging_dir'), cfg.get('logging_level'))


logger.info('Starting training procedure...')


run_ts = datetime.now().strftime('%Y%m%d%H%M%S')


if cfg.get('log_to_neptune'):
    logger.info('Initializing Neptune logging...')
    run = neptune.init(
        project=os.path.join(cfg.get('neptune_username'), cfg.get('neptune_project')),
        api_token=cfg.get('neptune_api_token')
    )
else:
    run = None


model = LitHMAutoEncoder(
    data_path=cfg.get('data_path'),
    batch_size=cfg.get('batch_size'),
    encoder=getattr(import_module('model.encoders'), cfg.get('encoder'))(cfg.get('embedding_size')),
    decoder=getattr(import_module('model.decoders'), cfg.get('decoder'))(cfg.get('embedding_size')),
    num_workers=cfg.get('num_workers'),
    center=cfg.get('center'),
    center_params=cfg.get('center_params'),
    optimizer=getattr(import_module('torch.optim'), cfg.get('optimizer')),
    optimizer_params=cfg.get('optimizer_params'),
    run=run
)


if not os.path.exists(cfg.get('output_path')):
    os.makedirs(cfg.get('output_path'))
checkpoint_path = os.path.join(cfg.get('output_path'), 'model' + run_ts)
os.makedirs(checkpoint_path)


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=checkpoint_path,
    monitor='val_loss',
    filename='model-{epoch:02d}-{val_loss:.4f}',
)


trainer = pl.Trainer(
    max_epochs=cfg.get('num_epochs'),
    gpus=cfg.get('num_gpus'),
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback]
)


settings_record = {
    'data_folder': os.path.basename(model.data_path),
    'center': model.center,
    'center_params': str(model.center_params) if model.center else None,
    'batch_size': model.batch_size,
    'train_valid_ratio': model.train_valid_ratio,
    'num_epochs': trainer.max_epochs,
    'num_workers': model.num_workers,
    'num_gpus': trainer.gpus,
    'optimizer': cfg.get('optimizer'),
    'optimizer_params': str(model.optimizer_params),
    'embedding_size': cfg.get('embedding_size'),
    'encoder': cfg.get('encoder'),
    'decoder': cfg.get('decoder'),
    'model_id': run_ts
}
if cfg.get('log_to_neptune'):
    run['settings'] = settings_record


trainer.fit(model)


logger.info('All done.')
