import torch
from typing import List
from abc import ABC, abstractmethod

class TextProcess(ABC):
    aux_vocab = ["<p>", "<s>", "<e>", " ", ":", "'"] + list(map(str, range(10)))

    origin_list_vocab = {
        "en": aux_vocab + list("abcdefghijklmnopqrstuvwxyz"),
        "vi": aux_vocab
        + list(
            "abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
        ),
    }

    origin_vocab = {
        lang: dict(zip(vocab, range(len(vocab))))
        for lang, vocab in origin_list_vocab.items()
    }

    def __init__(self, lang: str = "vi", **kwargs):
        super().__init__()
        self.lang = lang
        assert self.lang in ["vi", "en"], "Language not found"
        self.vocab = self.origin_vocab[lang]
        self.list_vocab = self.origin_list_vocab[lang]
        self.n_class = len(self.list_vocab)
        self.sos_id = 1
        self.eos_id = 2
        self.blank_id = 0

    def tokenize(self, s: str) -> List:
        return list(s)

    def decode(self, argmax: torch.Tensor):
        """
        decode greedy with collapsed repeat
        """
        decode = []
        for i, index in enumerate(argmax):
            if index != self.blank_id:
                if i != 0 and index == argmax[i - 1]:
                    continue
                decode.append(index.item())
        return self.int2text(decode)

    def text2int(self, s: str) -> torch.Tensor:
        return torch.Tensor([self.vocab[i] for i in s])

    def int2text(self, s: torch.Tensor) -> str:
        text = ""
        for i in s:
            if i in [self.sos_id, self.blank_id]:
                continue
            if i == self.eos_id:
                break
            text += self.list_vocab[i]
        return text