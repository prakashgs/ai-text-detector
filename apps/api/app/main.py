from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .gptzero import analyze_gptzero
from .local_detectors import analyze_detectgpt, analyze_gpt2

app = FastAPI(title="AI text detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)


class DetectorResult(BaseModel):
    method: str
    perplexity: Optional[float] = None
    burstiness: Optional[float] = None
    score: Optional[float] = None
    score_label: str
    detail: str
    error: Optional[str] = None


def _failed(method: str, score_label: str, exc: Exception) -> dict:
    return {
        "method": method,
        "perplexity": None,
        "burstiness": None,
        "score": None,
        "score_label": score_label,
        "detail": "This detector did not finish.",
        "error": str(exc),
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/analyze", response_model=list[DetectorResult])
def analyze(body: AnalyzeRequest) -> list[dict]:
    text = " ".join(body.text.split())
    if len(text) < 40:
        raise HTTPException(status_code=400, detail="Enter at least 40 characters.")
    results: list[dict] = []
    for runner, method, label in (
        (analyze_gpt2, "GPT-2", "Predictability"),
        (analyze_detectgpt, "DetectGPT", "Curvature"),
    ):
        try:
            results.append(runner(text))
        except Exception as exc:  # noqa: BLE001 — one detector should not blank the others
            results.append(_failed(method, label, exc))
    try:
        results.append(analyze_gptzero(text))
    except Exception as exc:  # noqa: BLE001
        results.append(_failed("GPTZero", "AI probability", exc))
    return results
