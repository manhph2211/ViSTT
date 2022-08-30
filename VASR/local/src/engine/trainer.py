import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Dict, Tuple
from src.losses.ctc import CTCLoss
from src.utils.utils import TextProcess
from src.metrics.wer import cal_wer


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.cfg_model.optim.adamw)
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer, **self.cfg_model.lr_scheduler.one_cycle_lr
            ),
            "name": "lr_scheduler_logger",
        }
        return [optimizer], [lr_scheduler]

    def get_wer(
        self, targets: Tensor, inputs: Tensor, input_lengths: Tensor
    ) -> Tuple[List[str], List[str], float]:
        predict_sequences = self.recognize(inputs, input_lengths)
        label_sequences = list(map(self.text_process.int2text, targets))
        wer = torch.Tensor(
            [
                cal_wer(truth, hypot)
                for truth, hypot in zip(label_sequences, predict_sequences)
            ]
        )
        wer = torch.mean(wer).item()
        return label_sequences, predict_sequences, wer

    def log_output(self, predict, target, wer):
        print("=" * 50)
        print("Sample Predicts: ", predict)
        print("Sample Targets:", target)
        print("Mean WER:", wer)
        print("=" * 50)


class ConformerModule(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        n_class: int,
        cfg_model: Dict,
        text_process: TextProcess,
        log_idx: int = 100,
    ):
        super().__init__()
        self.encoder = encoder
        self.out = nn.Linear(encoder.output_dim, n_class)
        self.criterion = CTCLoss(**cfg_model.loss.ctc)

        self.cfg_model = cfg_model
        self.text_process = text_process
        self.log_idx = log_idx
        self.save_hyperparameters()

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.out(outputs)
        outputs = F.log_softmax(outputs, -1)
        return outputs, output_lengths

    @torch.no_grad()
    def decode(self, encoder_output: Tensor) -> str:
        encoder_output = encoder_output.unsqueeze(0)
        outputs = F.log_softmax(self.out(encoder_output), -1)
        argmax = outputs.squeeze(0).argmax(-1)
        return self.text_process.decode(argmax)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> List[str]:
        outputs = list()

        encoder_outputs, _ = self.encoder(inputs, input_lengths)

        for encoder_output in encoder_outputs:
            predict = self.decode(encoder_output)
            outputs.append(predict)

        return outputs

    def training_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch

        targets_ctc = targets[:, 1:-1]

        outputs, output_lengths = self(inputs, input_lengths)

        loss = self.criterion(
            outputs.permute(1, 0, 2), targets_ctc, output_lengths, target_lengths
        )

        self.log("train loss", loss)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch

        targets_ctc = targets[:, 1:-1]

        outputs, output_lengths = self(inputs, input_lengths)

        loss = self.criterion(
            outputs.permute(1, 0, 2), targets_ctc, output_lengths, target_lengths
        )

        self.log("test loss", loss)

        if batch_idx % self.log_idx == 0:
            label_sequences, predict_sequences, wer = self.get_wer(
                targets_ctc, inputs, input_lengths
            )
            self.log_output(predict_sequences[0], label_sequences[0], wer)
            self.log("test wer", wer)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch

        targets_ctc = targets[:, 1:-1]

        outputs, output_lengths = self(inputs, input_lengths)

        loss = self.criterion(
            outputs.permute(1, 0, 2), targets_ctc, output_lengths, target_lengths
        )

        self.log("test loss", loss)

        if batch_idx % self.log_idx == 0:
            label_sequences, predict_sequences, wer = self.get_wer(
                targets_ctc, inputs, input_lengths
            )
            self.log_output(predict_sequences[0], label_sequences[0], wer)
            self.log("test wer", wer)

        return loss

