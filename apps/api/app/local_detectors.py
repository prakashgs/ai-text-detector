from __future__ import annotations

import random
import re

import torch

from .metrics import population_variance, safe_perplexity
from .models import bundle
from .scoring import document_perplexity, mean_nll, sentence_burstiness

PERTURBATIONS = 8
MASK_FRACTION = 0.15
SPAN_LENGTH = 2


def _mask_words(words: list[str]) -> tuple[str, list[tuple[int, int]]]:
    count = len(words)
    span = SPAN_LENGTH if count >= 12 else 1
    n_spans = max(1, int(round(MASK_FRACTION * count / span)))
    order = list(range(count))
    random.shuffle(order)
    used: set[int] = set()
    spans: list[tuple[int, int]] = []
    for start in order:
        if len(spans) >= n_spans:
            break
        end = start + span
        if end > count or any(index in used for index in range(start, end)):
            continue
        spans.append((start, end))
        used.update(range(start, end))
    spans.sort()
    pieces: list[str] = []
    cursor = 0
    for index, (start, end) in enumerate(spans):
        if cursor < start:
            pieces.append(" ".join(words[cursor:start]))
        pieces.append(f"<extra_id_{index}>")
        cursor = end
    if cursor < count:
        pieces.append(" ".join(words[cursor:]))
    return " ".join(piece for piece in pieces if piece), spans


def _fills(generated: str) -> dict[int, str]:
    cleaned = generated.replace("</s>", " ").replace("<pad>", " ")
    parts = re.split(r"<extra_id_(\d+)>", cleaned)
    fills: dict[int, str] = {}
    if len(parts) < 3:
        return fills
    iterator = iter(parts[1:])
    for raw_index, chunk in zip(iterator, iterator):
        words = chunk.strip().split()
        if words:
            fills[int(raw_index)] = " ".join(words)
    return fills


def _stitch(words: list[str], spans: list[tuple[int, int]], generated: str) -> str | None:
    fills = _fills(generated)
    if not fills:
        return None
    rewritten = words[:]
    for index, (start, end) in reversed(list(enumerate(spans))):
        replacement = fills.get(index, "").split()
        if not replacement:
            return None
        rewritten[start:end] = replacement
    return " ".join(rewritten)


def _perturb_once(words: list[str]) -> str | None:
    masked, spans = _mask_words(words)
    model, tokenizer = bundle.t5()
    encoded = tokenizer(masked, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        output = model.generate(
            encoded.input_ids.to(bundle.device),
            attention_mask=encoded.attention_mask.to(bundle.device),
            do_sample=True,
            top_p=0.96,
            temperature=1.0,
            max_new_tokens=48,
        )
    generated = tokenizer.decode(output[0], skip_special_tokens=False)
    return _stitch(words, spans, generated)


def analyze_detectgpt(text: str) -> dict:
    words = text.split()
    if len(words) < 8:
        raise ValueError("DetectGPT needs at least 8 words so spans can be masked.")
    original_nll = mean_nll(text)
    perturbed_nlls: list[float] = []
    perturbed_ppls: list[float] = []
    for _ in range(PERTURBATIONS):
        rewritten = _perturb_once(words)
        if not rewritten or rewritten == text:
            continue
        try:
            nll = mean_nll(rewritten)
        except ValueError:
            continue
        perturbed_nlls.append(nll)
        perturbed_ppls.append(safe_perplexity(nll))
    if len(perturbed_nlls) < 2:
        raise RuntimeError("Not enough successful perturbations to score DetectGPT.")
    mean_nll_perturbed = sum(perturbed_nlls) / len(perturbed_nlls)
    variance = population_variance(perturbed_nlls) or 0.0
    std = variance ** 0.5
    curvature = (mean_nll_perturbed - original_nll) / std if std > 1e-6 else 0.0
    burstiness = population_variance(perturbed_ppls)
    return {
        "method": "DetectGPT",
        "perplexity": sum(perturbed_ppls) / len(perturbed_ppls),
        "burstiness": burstiness,
        "score": curvature,
        "score_label": "Curvature",
        "detail": (
            f"{len(perturbed_ppls)} T5 fills. Chart point is the perturbed neighborhood "
            "(mean perplexity, variance of those perplexities). "
            f"Original GPT-2 perplexity is {document_perplexity(text):.1f}."
        ),
        "error": None,
    }


def analyze_gpt2(text: str) -> dict:
    perplexity = document_perplexity(text)
    burstiness, sentence_count = sentence_burstiness(text)
    # Lower perplexity is more predictable. Map a rough 0-1 "machine-like" hint for the table only.
    machine_hint = max(0.0, min(1.0, (80 - perplexity) / 80))
    detail = f"Document perplexity under GPT-2. Burstiness uses {sentence_count} sentences."
    if sentence_count < 2:
        detail += " Add another sentence to get a burstiness variance."
    return {
        "method": "GPT-2",
        "perplexity": perplexity,
        "burstiness": burstiness,
        "score": machine_hint,
        "score_label": "Predictability",
        "detail": detail,
        "error": None,
    }
