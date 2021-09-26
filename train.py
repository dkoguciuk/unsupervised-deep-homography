import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate

import argparse
import pytorch_lightning as pl

from dataset import SyntheticDataset, safe_collate
import model


class HomographyModel(pl.LightningModule):
    def __init__(self, hparams=None):
        super(HomographyModel, self).__init__()
        if hparams is not None:
            self.save_hyperparameters(hparams)
            if type(hparams) != dict:
                hparams = vars(hparams)
            self.loss_fn = getattr(model, hparams['loss'])
        self.model = model.Net()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = self.loss_fn(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        self.log('loss', loss)
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = self.loss_fn(delta, img_a, patch_b, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        train_set = SyntheticDataset(self.hparams.train_path, rho=self.hparams.rho)
        return DataLoader(
            train_set,
            num_workers=4,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=safe_collate,
        )

    def val_dataloader(self):
        val_set = SyntheticDataset(self.hparams.valid_path, rho=self.hparams.rho)
        return DataLoader(
            val_set,
            num_workers=4,
            batch_size=self.hparams.batch_size,
            collate_fn=safe_collate,
        )


def main(args):
    if args.resume:
        model = HomographyModel.load_from_checkpoint(args.resume)
    else:
        model = HomographyModel(hparams=args)
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, log_every_n_steps=100, flush_logs_every_n_steps=100)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--rho", type=int, default=45, help="amount to perturb corners")
    parser.add_argument("--resume", type=str, help="checkpoint to resume from", default="")
    parser.add_argument("--loss", type=str, help="loss used", default="photometric_loss")
    parser.add_argument("train_path", help="path to training imgs")
    parser.add_argument("valid_path", help="path to validation imgs")
    args = parser.parse_args()
    main(args)
