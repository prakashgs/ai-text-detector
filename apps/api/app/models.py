from __future__ import annotations

import threading

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

GPT2_NAME = "gpt2"
T5_NAME = "t5-small"


class ModelBundle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gpt2 = None
        self._gpt2_tokenizer = None
        self._t5 = None
        self._t5_tokenizer = None
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def gpt2(self):
        with self._lock:
            if self._gpt2 is None:
                self._gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_NAME)
                self._gpt2 = AutoModelForCausalLM.from_pretrained(GPT2_NAME).to(self.device)
                self._gpt2.eval()
            return self._gpt2, self._gpt2_tokenizer

    def t5(self):
        with self._lock:
            if self._t5 is None:
                self._t5_tokenizer = AutoTokenizer.from_pretrained(T5_NAME)
                self._t5 = AutoModelForSeq2SeqLM.from_pretrained(T5_NAME).to(self.device)
                self._t5.eval()
            return self._t5, self._t5_tokenizer


bundle = ModelBundle()
