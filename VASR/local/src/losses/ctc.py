from torch import nn, Tensor
from torchaudio import transforms as T


class CTCLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.CTCLoss(**kwargs)

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        return self.loss(log_probs, targets, input_lengths, target_lengths)

