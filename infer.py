from utils.configuration import load_config
from utils.logger import configure_logger
from model.autoencoder_module import LitHMAutoEncoder
from importlib import import_module
from datetime import datetime
import os
from data.hm_data import save_embeddings


def main(run_ts):

    cfg = load_config('config.yml')
    logger = configure_logger(__name__, cfg.get('logging_dir'), cfg.get('logging_level'))

    logger.info('Starting inference procedure...')

    logger.info(f'Loading model from: {cfg.get("infer_model_path")}')
    infer_model_fpath = os.path.join(cfg.get('infer_model_path'), os.listdir(cfg.get('infer_model_path'))[0])

    model = LitHMAutoEncoder.load_from_checkpoint(
        infer_model_fpath,
        batch_size=cfg.get('batch_size'),
        encoder=getattr(import_module('model.encoders'), cfg.get('encoder'))(cfg.get('embedding_size')),
        decoder=getattr(import_module('model.decoders'), cfg.get('decoder'))(cfg.get('embedding_size')),
        num_workers=cfg.get('num_workers')
    )

    embeddings, article_ids = model.calculate_embeddings(cfg.get('infer_data_path'))
    output_path = os.path.join(
        cfg.get('output_path'),
        'embeddings' + run_ts + '_' + os.path.basename(cfg.get('infer_model_path'))
    )
    os.makedirs(output_path)

    save_embeddings(
        embeddings,
        output_path,
        article_ids=article_ids,
        to_parquet=cfg.get('save_as_parquet')
    )

    logger.info('All done.')


if __name__ == '__main__':
    run_ts = datetime.now().strftime('%Y%m%d%H%M%S')
    main(run_ts=run_ts)
