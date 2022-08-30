import torch
import torchaudio.transforms as T
import pytorch_lightning as pl
import torchmetrics
from src.utils.utils import TextProcess
from src.models.conformer.conformer import Conformer
from transformers import get_linear_schedule_with_warmup


class ConformerModule(pl.LightningModule):
    def __init__(
        self, cfg, blank: int = 0, text_process: TextProcess = None, batch_size: int = 4
    ):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.training.lr
        self.conformer = Conformer(**cfg.model)
        self.cal_loss = T.RNNTLoss(blank=blank)
        self.cal_wer = torchmetrics.WordErrorRate()
        self.text_process = text_process
        self.save_hyperparameters()

    def forward(self, inputs, input_lengths):
        return self.conformer.recognize(inputs, input_lengths)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, **self.cfg.optim)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, **self.cfg.sched)
        return [optimizer], [lr_scheduler]

    def get_batch(self, batch):
        inputs, input_lengths, targets, target_lengths = batch

        batch_size = inputs.size(0)

        zeros = torch.zeros((batch_size, 1)).to(device=self.device)
        compute_targets = torch.cat((zeros, targets), dim=1).to(
            device=self.device, dtype=torch.int
        )
        compute_target_lengths = (target_lengths + 1).to(device=self.device)

        return (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        )

    def training_step(self, batch, batch_idx):
        (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        ) = self.get_batch(batch)

        outputs, output_lengths = self.conformer(
            inputs, input_lengths, compute_targets, compute_target_lengths
        )

        loss = self.cal_loss(outputs, targets, output_lengths, target_lengths)
        self.log("train_loss", loss)
        self.log("lr", self.lr)

        return loss

    def validation_step(self, batch, batch_idx):
        (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        ) = self.get_batch(batch)

        outputs, output_lengths = self.conformer(
            inputs, input_lengths, compute_targets, compute_target_lengths
        )

        loss = self.cal_loss(outputs, targets, output_lengths, target_lengths)

        predicts = self.forward(inputs, input_lengths)
        predicts = [self.text_process.int2text(sent) for sent in predicts]
        targets = [self.text_process.int2text(sent) for sent in targets]
        list_wer = torch.tensor(
            [self.cal_wer(i, j).item() for i, j in zip(predicts, targets)]
        )
        wer = torch.mean(list_wer)
        

        if batch_idx % 100 == 0:
            self.log_output(predicts[0], targets[0], wer)

        self.log("val_loss", loss)
        self.log("val_batch_wer", wer)

        return loss, wer

    def test_step(self, batch, batch_idx):
        (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        ) = self.get_batch(batch)

        outputs, output_lengths = self.conformer(
            inputs, input_lengths, compute_targets, compute_target_lengths
        )

        loss = self.cal_loss(outputs, targets, output_lengths, target_lengths)

        predicts = self.forward(inputs, input_lengths)
        predicts = [self.text_process.int2text(sent) for sent in predicts]
        targets = [self.text_process.int2text(sent) for sent in targets]

        list_wer = torch.tensor(
            [self.cal_wer(i, j).item() for i, j in zip(predicts, targets)]
        )
        wer = torch.mean(list_wer)

        if batch_idx % 100 == 0:
            self.log_output(predicts[0], targets[0], wer)

        self.log("test_loss", loss)
        self.log("test_batch_wer", wer)

        return loss, wer

    def log_output(self, predict, target, wer):
        print("=" * 50)
        print("Sample Predicts: ", predict)
        print("Sample Targets:", target)
        print("Mean WER:", wer)
        print("=" * 50)