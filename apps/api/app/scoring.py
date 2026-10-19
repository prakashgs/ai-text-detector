from __future__ import annotations

import torch

from .metrics import population_variance, safe_perplexity, split_sentences
from .models import bundle


def mean_nll(text: str) -> float:
    model, tokenizer = bundle.gpt2()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoded.input_ids.to(bundle.device)
    if input_ids.shape[1] < 2:
        raise ValueError("Text is too short to score.")
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return float(loss.item())


def document_perplexity(text: str) -> float:
    return safe_perplexity(mean_nll(text))


def sentence_burstiness(text: str) -> tuple[float | None, int]:
    scored: list[float] = []
    for sentence in split_sentences(text):
        if len(sentence.split()) < 4:
            continue
        try:
            scored.append(document_perplexity(sentence))
        except ValueError:
            continue
    return population_variance(scored), len(scored)
