from pathlib import Path

# PyTorch Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


class Rotor:
    def __init__(
            self,
            model_name,
            logging_params,
            checkpoint_callback_params,
            trainer_params
    ):
        super(Rotor).__init__()

        self.model_name = model_name
        self.logging_params = logging_params
        self.checkpoint_callback_params = checkpoint_callback_params
        self.trainer_params = trainer_params

    def __get_checkpoint_callback(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_callback_params['dirpath'],
            filename='{epoch}-{val_loss:.3f}',
            monitor=self.checkpoint_callback_params['monitor'],
            verbose=self.checkpoint_callback_params['verbose'],
            mode=self.checkpoint_callback_params['mode'],
            save_top_k=self.checkpoint_callback_params['save_top_k'],
        )

        return checkpoint_callback

    def __get_monitor(self):
        lr_monitor = LearningRateMonitor()
        return lr_monitor

    def __get_logger(self):
        logger = TensorBoardLogger(
            save_dir=str(Path(self.logging_params['save_dir'])),
            name=self.model_name
        )

        return logger

    def get_trainer(
            self,
            checkpoint_callback,
            lr_monitor,
            logger,
    ):
        trainer = Trainer(
            callbacks=[checkpoint_callback, lr_monitor],
            logger=logger,
            gpus=self.trainer_params['gpus'],
            max_epochs=self.trainer_params['max_epochs'],
            num_sanity_val_steps=self.trainer_params['num_sanity_val_steps']
        )

        return trainer

    def __call__(self):
        checkpoint_callback = self.__get_checkpoint_callback()
        tb_logger = self.__get_logger()
        lr_monitor = self.__get_monitor()

        trainer = self.get_trainer(
            checkpoint_callback=checkpoint_callback,
            lr_monitor=lr_monitor,
            logger=tb_logger
        )

        return trainer
