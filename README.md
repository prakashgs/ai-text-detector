# AI text detector

Simple app to determine if text is AI-generated or not.
Text is scored by GPT-2, a DetectGPT-style check, and GPTZero. The chart compares 2 metrics: perplexity with burstiness variance.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r apps/api/requirements.txt
cp .env.example .env   # optional GPTZero key
npm install
npm run dev
```

- Web: http://localhost:5173
- API: http://127.0.0.1:8000

The first analysis downloads `gpt2` and `t5-small`. DetectGPT runs several T5 rewrites, so a paragraph can take about a minute on CPU.

## What each point means

| Method | Perplexity (x) | Burstiness (y) | Score |
| --- | --- | --- | --- |
| GPT-2 | Perplexity of the passage | Variance of sentence perplexities | Predictability, higher when the passage is easier for GPT-2 |
| DetectGPT | Mean perplexity of T5 rewrites | Variance of those rewrite perplexities | Curvature: how much more surprising the rewrites are than the original |
| GPTZero | Mean of GPTZero sentence perplexities, when returned | Variance of those sentence perplexities | GPTZero AI probability |

Set `GPTZERO_API_KEY` in `.env`. Without it, GPT-2 and DetectGPT still run and GPTZero returns a clear error.
