import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from model import Encoder, Decoder, ResNet
import pytorch_lightning as pl
import pandas as pd


class EncoderResNetDecoder(pl.LightningModule):
    def __init__(self,  learning_rate):
        super().__init__()
        self.encoder = Encoder()
        self.resnet = ResNet(in_channels=256, num_blocks=5)
        self.decoder = Decoder()
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.encoder(x)
        x = self.resnet(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.forward(x)
        loss = F.mse_loss(z, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.forward(x)
        loss = F.mse_loss(z, y)
        self.log('val_loss', loss)


class EMG2HandPoseDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csv_dataset = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return (len(self.csv_dataset)  // 5) - (1000//5)

    def __getitem__(self, idx):
        subsequence = self.csv_dataset.iloc[idx*5:idx*5+1000]

        if self.transform:
            sample = self.transform(subsequence)

        x = subsequence[['Channel_1', 'Channel_2', 'Channel_3', 'Channel_4', 'Channel_5', 'Channel_6', 'Channel_7', 'Channel_8']].values
        y = subsequence[['Thumb_TMC_fe', 'Thumb_tmc_aa', 'Thumb_mcp_fe', 'Thumb_mcp_aa', 'Index_mcp_fe', 'Index_mcp_aa',
           'Index_pip', 'Middle_mcp_fe', 'Middle_mcp_aa', 'Middle_pip',
           'Ring_mcp_fe', 'Ring_mcp_aa', 'Ring_pip', 'Little_mcp_fe',
           'Little_mcp_aa', 'Little_pip']].values

        return torch.from_numpy(x.astype('float32')).unsqueeze(0), torch.from_numpy(y.astype('float32')).unsqueeze(0)


dataset = EMG2HandPoseDataset("./core_angles-core_hand-90s.csv")

train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4) #TODO : create a validation dataset and dataloader

model = EncoderResNetDecoder(learning_rate=0.001)

# training
trainer = pl.Trainer(auto_lr_find=True)
trainer.tune(model, train_dataloader, val_dataloader)
trainer.fit(model, train_dataloader, val_dataloader)