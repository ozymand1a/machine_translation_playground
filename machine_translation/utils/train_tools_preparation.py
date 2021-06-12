from pathlib import Path

# PyTorch Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class Rotor():
    def __init__(self,
                 model_name,
                 general_params,
                 checkpoint_callback_params,
                 logger_params,
                 train_params):
        super(Rotor).__init__()

        self.model_name = model_name
        self.general_params = general_params
        self.checkpoint_callback_params = checkpoint_callback_params
        self.logger_params = logger_params
        self.train_params = train_params

    def get_checkpoint_callback(self):
        checkpoint_callback = ModelCheckpoint(
            filename=str(Path(self.general_params['logs_dir']).joinpath(self.model_name).joinpath('weights')),
            verbose=self.checkpoint_callback_params['verbose'],
            monitor=self.checkpoint_callback_params['monitor'],
            mode=self.checkpoint_callback_params['mode'],
            save_top_k=self.checkpoint_callback_params['save_top_k'],
        )

        return checkpoint_callback

    def get_logger(self):
        logger = TensorBoardLogger(
            str(Path(self.general_params['logs_dir'])),
            name=self.model_name
        )

        return logger

    def get_trainer(self, checkpoint_callback, logger):
        trainer = Trainer(
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            gpus=[self.general_params['ngpu']] if self.general_params['cuda'] else None,
            log_gpu_memory='all',
            max_epochs=self.train_params['n_epochs'],
            num_sanity_val_steps=self.train_params['batch_size']
        )

        return trainer

    def __call__(self):
        checkpoint_callback = self.get_checkpoint_callback()
        tb_logger = self.get_logger()

        trainer = self.get_trainer(checkpoint_callback, tb_logger)
        return trainer
