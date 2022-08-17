
import random
import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).
    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment.
        Can be a positive integer or a float value in the range [0, 1].
        If positive integer value, defines maximum number of time steps
        to be cut in one segment.
        If a float value, defines maximum percentage of timesteps that
        are cut adaptively.
    """

    def __init__(
        self,
        freq_masks=0,
        time_masks=0,
        freq_width=10,
        time_width=10,
        rng=None,
        mask_value=0.0,
    ):
        super().__init__()

        self._rng = random.Random() if rng is None else rng

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

        self.mask_value = mask_value

        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError(
                    "If `time_width` is a float value, must be in range [0, 1]"
                )

            self.adaptive_temporal_width = True

    @torch.no_grad()
    def forward(self, input_spec, length):
        sh = input_spec.shape

        for idx in range(sh[0]):
            for i in range(self.freq_masks):
                x_left = self._rng.randint(0, sh[1] - self.freq_width)

                w = self._rng.randint(0, self.freq_width)

                input_spec[idx, x_left : x_left + w, :] = self.mask_value

            for i in range(self.time_masks):
                if self.adaptive_temporal_width:
                    time_width = max(1, int(length[idx] * self.time_width))
                else:
                    time_width = self.time_width

                y_left = self._rng.randint(0, max(1, length[idx] - time_width))

                w = self._rng.randint(0, time_width)

                input_spec[idx, :, y_left : y_left + w] = self.mask_value

        return input_spec
