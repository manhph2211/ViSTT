import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from src.utils.utils import TextProcess


class VivosDataset(Dataset):
    def __init__(self, root: str = "", subset: str = "train", n_fft: int = 200):
        super().__init__()
        self.root = root
        self.subset = subset
        assert self.subset in ["train", "test"], "subset not found"

        path = os.path.join(self.root, self.subset)
        waves_path = os.path.join(path, "waves")
        transcript_path = os.path.join(path, "prompts.txt")

        # walker oof
        self.walker = list(Path(waves_path).glob("*/*"))

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcripts = f.read().strip().split("\n")
            transcripts = [line.split(" ", 1) for line in transcripts]
            filenames = [i[0] for i in transcripts]
            trans = [i[1] for i in transcripts]
            self.transcripts = dict(zip(filenames, trans))

        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        filepath = str(self.walker[idx])
        filename = filepath.rsplit(os.sep, 1)[-1].split(".")[0]

        wave, sr = torchaudio.load(filepath)
        specs = self.feature_transform(wave)  # channel, feature, time
        specs = specs.permute(0, 2, 1)  # channel, time, feature
        specs = specs.squeeze()  # time, feature

        trans = self.transcripts[filename].lower()

        return specs, trans


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

    def tokenize(self, s):
        s = s.lower()
        s = self.text_process.tokenize(s)
        return s

    def _collate_fn(self, batch):
        """
        Take feature and input, transform and then padding it
        """

        specs = [i[0] for i in batch]
        input_lengths = torch.IntTensor([i.size(0) for i in specs])
        trans = [i[1] for i in batch]

        bs = len(specs)

        # batch, time, feature
        specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)

        trans = [self.text_process.text2int(self.tokenize(s)) for s in trans]
        target_lengths = torch.IntTensor([s.size(0) for s in trans])
        trans = torch.nn.utils.rnn.pad_sequence(trans, batch_first=True).to(
            dtype=torch.int
        )

        # concat sos and eos to transcript
        sos_id = torch.IntTensor([[self.text_process.sos_id]]).repeat(bs, 1)
        eos_id = torch.IntTensor([[self.text_process.eos_id]]).repeat(bs, 1)
        trans = torch.cat((sos_id, trans, eos_id), dim=1).to(dtype=torch.int)

        return specs, input_lengths, trans, target_lengths