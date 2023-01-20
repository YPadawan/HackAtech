import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from model import Encoder, Decoder, ResNet
import pytorch_lightning as pl
import pandas as pd


class EncoderResNetDecoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.resnet = ResNet(in_channels=256, num_blocks=5)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.resnet(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)


# data
dataset = pd.read_csv("./core_angles-core_hand-90s.csv")
data = dataset.iloc[:, :8].values
y = df.iloc[:, 8:].values

data_pytorch = torch.from_numpy(data)
y_pytorch = torch.from_numpy(y)

#dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
data_train, data_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = EncoderResNetDecoder()

# training
trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)

