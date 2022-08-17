import torch
from torch import nn, Tensor
from .encoder import ConformerEncoder
from .decoder import DecoderRNNT


class Conformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        decoder_output_dim: int,
        hidden_state_dim: int = 320,
        decoder_num_layers: int = 1,
        input_dim: int = 80,
        num_heads: int = 4,
        encoder_dim: int = 144,
        num_layers: int = 16,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        subsampling_factor: int = 4,
        half_step_residual: bool = True,
        freq_masks: int = 2,
        time_masks: int = 10,
        freq_width: int = 27,
        time_width: float = 0.05,
        rnn_type: str = "lstm",
        sos_id: int = 1,
        eos_id: int = 2,
        grad_ckpt_batchsize: int = 4,
    ):
        super().__init__()
        self.grad_ckpt_batchsize = grad_ckpt_batchsize
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            num_heads=num_heads,
            encoder_dim=encoder_dim,
            num_layers=num_layers,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            subsampling_factor=subsampling_factor,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            freq_masks=freq_masks,
            time_masks=time_masks,
            freq_width=freq_width,
            time_width=time_width,
            grad_ckpt_batchsize=grad_ckpt_batchsize,
        )
        self.decoder = DecoderRNNT(
            num_classes=num_classes,
            hidden_state_dim=hidden_state_dim,
            output_dim=decoder_output_dim,
            num_layers=decoder_num_layers,
            rnn_type=rnn_type,
            sos_id=sos_id,
            eos_id=eos_id,
            dropout_p=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(encoder_dim << 1, encoder_dim),
            nn.Tanh(),
            nn.Linear(encoder_dim, num_classes, bias=False),
        )

    def set_encoder(self, encoder):
        """Setter for encoder"""
        self.encoder = encoder

    def set_decoder(self, decoder):
        """Setter for decoder"""
        self.decoder = decoder

    def count_parameters(self) -> int:
        """Count parameters of encoder"""
        num_encoder_parameters = self.encoder.count_parameters()
        num_decoder_parameters = self.decoder.count_parameters()
        return num_encoder_parameters + num_decoder_parameters

    def update_dropout(self, dropout_p) -> None:
        """Update dropout probability of model"""
        self.encoder.update_dropout(dropout_p)
        self.decoder.update_dropout(dropout_p)

    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        """
        Joint `encoder_outputs` and `decoder_outputs`.
        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.fc(outputs)

        return outputs

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs, _ = self.decoder(targets, target_lengths)

        outputs = self.joint(encoder_outputs, decoder_outputs)

        return outputs, encoder_output_lengths

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> Tensor:
        """
        Decode `encoder_outputs`.
        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens, hidden_state = list(), None
        decoder_input = encoder_output.new_tensor(
            [[self.decoder.sos_id]], dtype=torch.long
        )

        for t in range(max_length):

            decoder_output, hidden_state = self.decoder(
                decoder_input, hidden_states=hidden_state
            )
            step_output = self.joint(
                encoder_output[t].view(-1), decoder_output.view(-1)
            )
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)

        return torch.LongTensor(pred_tokens)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor):
        """
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        outputs = list()

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs