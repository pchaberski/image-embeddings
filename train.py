from utils.configuration import load_config
from utils.logger import configure_logger
import neptune.new as neptune
import os
from model.autoencoder_module import LitHMAutoEncoder
import pytorch_lightning as pl
from importlib import import_module
from datetime import datetime



def main(run_ts):

    cfg = load_config('config.yml')
    logger = configure_logger(__name__, cfg.get('logging_dir'), cfg.get('logging_level'))

    logger.info('Starting training procedure...')

    if cfg.get('log_to_neptune'):
        logger.info('Initializing Neptune logging...')
        run = neptune.init(
            project=cfg.get('neptune_username') + '/' + cfg.get('neptune_project'),
            api_token=cfg.get('neptune_api_token'),
            custom_run_id=run_ts
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
        lr_scheduler=getattr(import_module('torch.optim.lr_scheduler'), cfg.get('lr_scheduler')),
        lr_scheduler_params=cfg.get('lr_scheduler_params'),
        run=run
    )

    if not os.path.exists(cfg.get('output_path')):
        os.makedirs(cfg.get('output_path'))
    checkpoint_path = os.path.join(cfg.get('output_path'), 'model' + run_ts)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor='val_loss',
        filename='model-{epoch:02d}-{val_loss:.4f}',
    )

    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00001,
        patience=5,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=cfg.get('num_epochs'),
        gpus=cfg.get('num_gpus'),
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, early_stop_callback]
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
        'optimizer': cfg.get('lr_scheduler'),
        'optimizer_params': str(model.lr_scheduler_params),
        'embedding_size': cfg.get('embedding_size'),
        'encoder': cfg.get('encoder'),
        'decoder': cfg.get('decoder'),
        'num_params': model.get_num_params(),
        'model_id': run_ts
    }
    if cfg.get('log_to_neptune'):
        run['settings'] = settings_record

    trainer.fit(model)

    logger.info('All done.')


if __name__ == '__main__':
    run_ts = datetime.now().strftime('%Y%m%d%H%M%S')
    main(run_ts=run_ts)
