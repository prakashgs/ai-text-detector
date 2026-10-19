import { useState } from "react";
import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";

type DetectorResult = {
  method: string;
  perplexity: number | null;
  burstiness: number | null;
  score: number | null;
  score_label: string;
  detail: string;
  error: string | null;
};

const COLORS: Record<string, string> = {
  "GPT-2": "#8a5a2a",
  DetectGPT: "#1f6f8b",
  GPTZero: "#2f6b4f",
};

const SAMPLE = `The committee reviewed the proposal on Tuesday and asked for a clearer budget. Nobody expected the vote to happen before the holiday, but the chair pushed it through anyway. I still think the third paragraph overstates the risk. A model, by contrast, tends to keep every sentence equally smooth and equally sure of itself. That evenness is part of what these detectors are trying to measure.`;

function formatNumber(value: number | null, digits = 1): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function App() {
  const [text, setText] = useState(SAMPLE);
  const [results, setResults] = useState<DetectorResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = typeof payload?.detail === "string" ? payload.detail : "Analysis failed.";
        throw new Error(detail);
      }
      setResults(payload as DetectorResult[]);
    } catch (err) {
      setResults(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const points = (results ?? []).filter((row) => row.perplexity != null && row.burstiness != null);

  return (
    <div className="shell">
      <header>
        <p className="kicker">Perplexity and burstiness</p>
        <h1>AI text detector</h1>
        <p className="lede">
          One passage is scored three ways: GPT-2, a DetectGPT-style perturbation check, and GPTZero.
          The chart places each method by perplexity and burstiness variance.
        </p>
      </header>

      <form onSubmit={onSubmit}>
        <label htmlFor="passage">Passage</label>
        <textarea
          id="passage"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={10}
        />
        <div className="actions">
          <button type="submit" disabled={busy || text.trim().length < 40}>
            {busy ? "Scoring…" : "Score with all three"}
          </button>
          <button type="button" className="ghost" onClick={() => setText(SAMPLE)} disabled={busy}>
            Use sample
          </button>
        </div>
        <p className="hint">
          The first run downloads GPT-2 and T5-small. DetectGPT fills several masked spans, so a paragraph can take a minute.
        </p>
      </form>

      {error ? <p className="banner">{error}</p> : null}

      {results ? (
        <>
          <section>
            <h2>Perplexity vs burstiness</h2>
            <p className="hint">
              GPT-2 is the original passage. DetectGPT is the cloud of T5 rewrites (mean perplexity, variance across those rewrites). GPTZero uses its own sentence perplexities when the API returns them.
            </p>
            {points.length === 0 ? (
              <p className="hint">None of the detectors returned both perplexity and burstiness.</p>
            ) : (
              <div className="chart">
                <ResponsiveContainer width="100%" height={400}>
                  <ScatterChart margin={{ top: 28, right: 16, bottom: 28, left: 8 }}>
                    <CartesianGrid stroke="#e4d8c8" />
                    <XAxis
                      type="number"
                      dataKey="x"
                      name="Perplexity"
                      tick={{ fill: "#5c5146" }}
                      label={{ value: "Perplexity", position: "bottom", offset: 4, fill: "#5c5146" }}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      name="Burstiness"
                      tick={{ fill: "#5c5146" }}
                      label={{ value: "Burstiness variance", angle: -90, position: "insideLeft", fill: "#5c5146" }}
                    />
                    <ZAxis range={[120, 120]} />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(value: number) => formatNumber(value, 2)} />
                    <Legend verticalAlign="top" />
                    {points.map((row) => (
                      <Scatter
                        key={row.method}
                        name={row.method}
                        data={[{ x: row.perplexity, y: row.burstiness }]}
                        fill={COLORS[row.method] ?? "#333"}
                      />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            )}
          </section>

          <section>
            <h2>Scores</h2>
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Perplexity</th>
                  <th>Burstiness</th>
                  <th>Detector score</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {results.map((row) => (
                  <tr key={row.method}>
                    <td>{row.method}</td>
                    <td>{formatNumber(row.perplexity)}</td>
                    <td>{formatNumber(row.burstiness)}</td>
                    <td>
                      {row.score == null ? "—" : `${row.score_label} ${formatNumber(row.score, 3)}`}
                    </td>
                    <td>{row.error ?? row.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      ) : null}
    </div>
  );
}
