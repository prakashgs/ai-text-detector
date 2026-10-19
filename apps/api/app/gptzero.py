from __future__ import annotations

import os
from pathlib import Path

import httpx

from .metrics import population_variance

GPTZERO_URL = "https://api.gptzero.me/v2/predict/text"


def _load_env() -> None:
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def analyze_gptzero(text: str) -> dict:
    _load_env()
    api_key = os.environ.get("GPTZERO_API_KEY", "").strip()
    if not api_key:
        return {
            "method": "GPTZero",
            "perplexity": None,
            "burstiness": None,
            "score": None,
            "score_label": "AI probability",
            "detail": "Set GPTZERO_API_KEY in .env to call GPTZero.",
            "error": "GPTZERO_API_KEY is not set.",
        }
    response = httpx.post(
        GPTZERO_URL,
        headers={"x-api-key": api_key, "Accept": "application/json"},
        json={"document": text},
        timeout=45,
    )
    if response.status_code >= 400:
        return {
            "method": "GPTZero",
            "perplexity": None,
            "burstiness": None,
            "score": None,
            "score_label": "AI probability",
            "detail": "GPTZero rejected the request.",
            "error": f"GPTZero HTTP {response.status_code}: {response.text[:240]}",
        }
    payload = response.json()
    document = (payload.get("documents") or [payload])[0]
    sentences = document.get("sentences") or []
    sentence_ppls = [
        float(sentence["perplexity"])
        for sentence in sentences
        if isinstance(sentence, dict) and sentence.get("perplexity") is not None
    ]
    perplexity = document.get("perplexity")
    if perplexity is None and sentence_ppls:
        perplexity = sum(sentence_ppls) / len(sentence_ppls)
    burstiness = document.get("overall_burstiness")
    if sentence_ppls:
        burstiness = population_variance(sentence_ppls)
    class_probs = document.get("class_probabilities") or {}
    score = document.get("average_generated_prob")
    if score is None:
        score = class_probs.get("ai")
    predicted = document.get("predicted_class") or document.get("document_classification") or "unknown"
    return {
        "method": "GPTZero",
        "perplexity": float(perplexity) if perplexity is not None else None,
        "burstiness": float(burstiness) if burstiness is not None else None,
        "score": float(score) if score is not None else None,
        "score_label": "AI probability",
        "detail": f"GPTZero class: {predicted}. Burstiness is the variance of their sentence perplexities when those are returned.",
        "error": None,
    }
