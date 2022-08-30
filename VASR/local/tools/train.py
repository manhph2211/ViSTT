import argparse
from base64 import encode
from distutils.command.config import config

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from src.utils.utils import TextProcess
from src.datasets.dataset import VivosDataset, VivosDataModule
from src.engine.trainer import ConformerModule
from src.models.conformer import Conformer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Config path")
    parser.add_argument("-cp", help="config path")  # config path
    parser.add_argument("-cn", help="config name")  # config name

    args = parser.parse_args()

    @hydra.main(config_path=args.cp, config_name=args.cn)
    def main(cfg: DictConfig):

        text_process = TextProcess(**cfg.text_process)
        cfg.model.num_classes = len(text_process.vocab)

        train_set = VivosDataset(**cfg.datasets.vivos, subset="train")
        test_set = VivosDataset(**cfg.datasets.vivos, subset="test")

        dm = VivosDataModule(
            train_set, test_set, text_process, **cfg.datamodule.vivos
        )
        encoder = Conformer(**cfg.model.encoder.conformer)
        model = ConformerModule(
            encoder= encoder, n_class=cfg.model.num_classes, cfg_model = cfg.model, text_process=text_process,
        )

        tb_logger = pl.loggers.tensorboard.TensorBoardLogger(**cfg.trainer.tb_logger)

        trainer = pl.Trainer(logger=tb_logger, **cfg.trainer.hyper)

        if cfg.ckpt.have_ckpt:
            trainer.fit(model, datamodule=dm, ckpt_path=cfg.ckpt.ckpt_path)
        else:
            try:
                trainer.fit(model=model, datamodule=dm)
            except Exception as e:
                print(str(e))

        trainer.save_checkpoint("ckpts/conformer_rnnt.ckpt", weights_only=True)
        print("Testing model")
        if cfg.ckpt.have_ckpt:
            trainer.test(model, datamodule=dm, ckpt_path=cfg.ckpt.ckpt_path)
        else:
            trainer.test(model, datamodule=dm)

    main()
