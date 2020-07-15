from os import makedirs

# PyTorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# Typing
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.functional import Tensor
from typing import Dict, Tuple, List

# PyTorch Lightning
import pytorch_lightning as pl

# Output Folder for the files
path = './output_lightning'

# create output folder if doesn't exist
makedirs(path, exist_ok=True)

# shape of the image (gray)
img_shape = (1, 28, 28)


# Generator Model
class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x) -> Tensor:
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = torch.tanh(self.fc4(x))
        return x.view(x.shape[0], *img_shape)


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x) -> Tensor:
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.sigmoid(self.fc4(x))
        return x


# Lightning Module
class GAN(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super(GAN, self).__init__()

        self.hparams = hparams
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, x) -> Tensor:
        return self.discriminator(x)

    def loss_function(self, y_hat, y) -> Tensor:
        return nn.BCELoss()(y_hat, y)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List]:
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.4, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.4, 0.999))

        return [optimizer_G, optimizer_D], []

    def prepare_data(self) -> Dataset:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
        train_data = datasets.MNIST('./data',
                                    train=True,
                                    download=False,
                                    transform=transform)
        return train_data

    def train_dataloader(self) -> DataLoader:
        train_data = self.prepare_data()
        train_loader = DataLoader(train_data,
                                  batch_size=self.hparams.batch_size,
                                  shuffle=True)
        return train_loader

    def training_step(self, batch, batch_idx, optimizer_idx) -> Dict:
        real_images, _ = batch
        valid = torch.ones(real_images.size(0), 1)
        fake = torch.zeros(real_images.size(0), 1)
        criterion = self.loss_function

        if optimizer_idx == 0:
            gen_input = torch.randn(real_images.shape[0], 100)
            self.gen_images = self.generator(gen_input)

            g_loss = criterion(
                self(self.gen_images), valid)

            tqdm_dict = {'g_loss': g_loss}
            output = {
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
                'g_loss': g_loss
            }
            return output

        if optimizer_idx == 1:
            real_loss = criterion(
                self(real_images), valid)
            fake_loss = criterion(
                self(self.gen_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2.0

            tqdm_dict = {'d_loss': d_loss}
            output = {
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
                'd_loss': d_loss
            }
            return output

    def on_epoch_end(self) -> None:
        utils.save_image(self.gen_images.data[:25],
                         path + '/%d.png' % self.current_epoch,
                         nrow=5,
                         padding=0,
                         normalize=True)


if __name__ == "__main__":
    from argparse import ArgumentParser

    # Hyperparameters
    parser = ArgumentParser(description='GAN MNIST Example')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)

    args = parser.parse_args()

    # Model Initialization
    gan = GAN(hparams=args)

    # Model Training
    trainer = pl.Trainer.from_argparse_args(args,
                                            max_epochs=20,
                                            fast_dev_run=True)

    trainer.fit(gan)
