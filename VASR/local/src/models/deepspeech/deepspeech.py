from torchaudio.models import DeepSpeech
import torchmetrics


class DeepSpeechModule():
    def __init__(
        self,
        n_feature: int,
        n_hidden: int,
        dropout: float,
        n_class: int,
        lr: float,
        text_process,
        cfg_optim: dict,
    ):
        super().__init__()
        self.deepspeech = DeepSpeech(
            n_feature=n_feature, n_hidden=n_hidden, n_class=n_class, dropout=dropout
        )
        self.lr = lr
        self.text_process = text_process
        self.cal_wer = torchmetrics.WordErrorRate()
        self.cfg_optim = cfg_optim
        self.criterion = nn.CTCLoss(zero_infinity=True)

    def forward(self, inputs):
        """predicting function"""
        if len(inputs.size()) == 3:
            # add batch
            inputs = inputs.unsqueeze(0)
        outputs = self.deepspeech(inputs)
        decode = outputs.argmax(dim=-1)
        predicts = [self.text_process.decode(sent) for sent in decode]
        return predicts
