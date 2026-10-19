from __future__ import annotations

import math
import re


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def population_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def safe_perplexity(mean_nll: float) -> float:
    # Guard overflow on very surprising short strings.
    return math.exp(min(mean_nll, 20))
