import torch.nn as nn
from torch import Tensor
from base.ffw import FeedForwardModule
from base.att import ConvSubsampling
from base.att import MultiHeadedSelfAttentionModule
from base.conv import ConformerConvModule
from src.models.conformer.modules import ResidualConnectionModule


class ConformerBlock(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.
    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        num_heads: int = 4,
        encoder_dim: int = 144,
        num_layers: int = 16,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        subsampling_factor: int = 4,
        half_step_residual: bool = True,
        freq_masks: int = 2,
        time_masks: int = 10,
        freq_width: int = 27,
        time_width: float = 0.05,
        grad_ckpt_batchsize: int = 4,
    ):
        super().__init__()
        self.grad_ckpt_batchsize = grad_ckpt_batchsize

        self.conv_subsampling = ConvSubsampling(
            input_dim=input_dim,
            feat_out=encoder_dim,
            conv_channels=encoder_dim,
            subsampling_factor=subsampling_factor,
        )
        self.input_projection = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.Dropout(p=dropout)
        )
        module_list = [
            ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                feed_forward_dropout_p=dropout,
                attention_dropout_p=dropout,
                conv_dropout_p=dropout,
                conv_kernel_size=conv_kernel_size,
                half_step_residual=half_step_residual,
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.Sequential(*module_list)

    def forward(self, x, lengths):
        x, lengths = self.conv_subsampling(x, lengths)
        x = self.input_projection(x)
        x = self.layers(x)
        return x, lengths