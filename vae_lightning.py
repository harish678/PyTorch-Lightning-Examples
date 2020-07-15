# PyTorch packages
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms

# Typing
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
from torch.functional import Tensor
from typing import Any, Dict, Tuple

# PyTorch Lightning
import pytorch_lightning as pl


# Lightning Module
class VAE(pl.LightningModule):

    # model start
    def __init__(self, hparams) -> None:
        super(VAE, self).__init__()

        self.hparams = hparams
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x) -> Tuple[Tensor, Tensor]:
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z) -> Tensor:
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, z) -> Tensor:
        return self.decode(z)

    # model end

    # Loss Function
    def loss_function(self, recon_x, x, mu, logvar) -> Tensor:
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    # Optimizer
    def configure_optimizers(self) -> Optimizer:
        return optim.Adam(self.parameters(), self.hparams.learning_rate)

    # Train Dataloader
    def train_dataloader(self) -> DataLoader:
        train_data = datasets.MNIST('../data',
                                    train=True,
                                    download=True,
                                    transform=transforms.ToTensor())
        train_loader = DataLoader(train_data,
                                  batch_size=self.hparams.batch_size,
                                  shuffle=True)
        return train_loader

    # Valid Dataloader
    def val_dataloader(self) -> DataLoader:
        valid_data = datasets.MNIST('../data',
                                    train=False,
                                    download=True,
                                    transform=transforms.ToTensor())
        valid_loader = DataLoader(valid_data,
                                  batch_size=self.hparams.batch_size)
        return valid_loader

    # Training Loop
    def training_step(self, batch, batch_idx) -> Dict:
        x, _ = batch

        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)

        loss = self.loss_function(x_hat, x, mu, logvar)
        log = {'train_loss': loss}

        return {'loss': loss, 'log': log}

    # Validation Loop
    def validation_step(self, batch, batch_idx) -> Dict:
        x, _ = batch

        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)

        val_loss = self.loss_function(x_hat, x, mu, logvar)
        return {'val_loss': val_loss, 'x_hat': x_hat}

    # Hook at validation epoch end
    def validation_epoch_end(self, outputs) -> Dict:
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        x_hat = outputs[-1]['x_hat']

        grid = make_grid(x_hat)

        path = './' + str(self.current_epoch) + '.png'
        save_image(grid, path)

        self.logger.experiment.add_image('images', grid, 0)

        log = {'avg_log_loss': val_loss}
        return {'log': log, 'val_loss': log}


if __name__ == "__main__":
    import argparse

    # Hyperparameters
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()

    # Model Initialization
    vae = VAE(hparams=args)
    # Model Training
    trainer = pl.Trainer.from_argparse_args(args,
                                            max_epochs=10,
                                            fast_dev_run=False)
    trainer.fit(vae)
