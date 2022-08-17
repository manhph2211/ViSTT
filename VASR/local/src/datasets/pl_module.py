import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T
import torchaudio.functional as F
from src.models.conformer.conformer import Conformer
import pytorch_lightning as pl
import torchmetrics
import sys
from src.utils.utils import TextProcess


class VivosDataModule(pl.LightningDataModule):
    def __init__(
        self,
        trainset: Dataset,
        testset: Dataset,
        text_process: TextProcess,
        batch_size: int,
        num_workers: int = 8,
    ):
        super().__init__()

        self.trainset = trainset
        self.valset = testset
        self.testset = testset
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.text_process = text_process

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def _collate_fn(self, batch):
        """
        Take feature and input, transform and then padding it
        """
        specs = [i[0] for i in batch]
        input_lengths = torch.IntTensor([i.size(0) for i in specs])
        trans = [i[1] for i in batch]
        target_lengths = torch.IntTensor([len(s) for s in trans])

        # batch, time, feature
        specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)
        # specs = specs.unsqueeze(1)  # batch, channel, time, feature

        trans = [self.text_process.text2int(s) for s in trans]
        trans = torch.nn.utils.rnn.pad_sequence(trans, batch_first=True).to(
            dtype=torch.int
        )

        return specs, input_lengths, trans, target_lengths