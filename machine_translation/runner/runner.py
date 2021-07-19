# Basic
import numpy as np

# PyTorch
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

# PyTorch Lightning
import pytorch_lightning as pl


class S2SRunner(pl.LightningModule):
    def __init__(
            self,
            model,
            train_iterator,
            val_iterator,
            trg_pad_idx,
            clip,
            epoch_size,
            with_pad,
            n_gpu
    ):
        super(S2SRunner, self).__init__()
        self.model = model

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.clip = clip
        self.epoch_size = epoch_size

        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

        self.with_pad = with_pad

        self.n_gpu = n_gpu

    def forward(
            self,
            x
    ):
        return self.model(x)

    def training_step(
            self,
            batch,
            batch_idx
    ):
        if self.with_pad:
            src, src_len = batch.src
            trg = batch.trg

            output = self.model(src, src_len, trg)
        else:
            src = batch.src
            trg = batch.trg

            output = self.model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = self.criterion(output.to(self.n_gpu), trg)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.log(
            'loss',
            loss,
            on_step=True,
            on_epoch=True,
            logger=True
        )

        return {'loss': loss}

    def validation_step(
            self,
            batch,
            batch_idx
    ):
        if self.with_pad:
            src, src_len = batch.src
            trg = batch.trg

            output = self.model(src, src_len, trg, 0)
        else:
            src = batch.src
            trg = batch.trg

            output = self.model(src, trg, 0)

        output_dim = output.shape[-1]

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        loss = self.criterion(output.to(self.n_gpu), trg)

        self.log(
            'val_loss',
            loss
        )

        return {'val_loss': loss}

    def validation_epoch_end(
            self,
            outputs
    ):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters())
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.5,
            patience=4,
            verbose=True
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.train_iterator

    def val_dataloader(self):
        return self.val_iterator
