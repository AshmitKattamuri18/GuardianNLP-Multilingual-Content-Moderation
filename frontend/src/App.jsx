import { useEffect, useState } from "react";
import "./App.css";

const API_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const CLASSES = ["NORMAL", "OFFENSIVE", "HATE_SPEECH"];
const score = (value) => Number.isFinite(value) ? value.toFixed(4) : "--";
const confidencePercent = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "--";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [backendReady, setBackendReady] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let active = true;
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_URL}/health`);
        const health = response.ok ? await response.json() : null;
        if (active) setBackendReady(health?.ready === true);
      } catch {
        if (active) setBackendReady(false);
      }
    };
    checkHealth();
    const timer = setInterval(checkHealth, 15000);
    return () => { active = false; clearInterval(timer); };
  }, []);

const handleModerate = async () => {
  if (!text.trim()) { setResult({ action: "ERROR", message: "Please enter some text to moderate." }); return; }
  setBusy(true);
  setResult({ action: "ANALYZING", message: "Analyzing text with GuardianNLP..." });
  try {
    const response = await fetch(`${API_URL}/moderate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: text.trim() }) });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data?.detail === "string" ? data.detail : "Moderation request failed.");
    if (!data || !CLASSES.includes(data.predicted_label) || !Number.isFinite(data.confidence) || !["ALLOW", "WARNING", "CENSOR", "BLOCK"].includes(data.action) || !Array.isArray(data.ensemble_probabilities) || data.ensemble_probabilities.length !== 3 || !data.ensemble_probabilities.every(Number.isFinite) || Math.abs(data.ensemble_probabilities.reduce((sum, value) => sum + value, 0) - 1) > 1e-4) throw new Error("The moderation API returned an invalid response.");
    setResult({ ...data, message: "Text successfully analyzed by GuardianNLP trained models." });
  } catch (error) {
    setResult({ action: "ERROR", message: error.message || "Unable to connect to the GuardianNLP backend." });
  } finally { setBusy(false); }
};

const getActionClass = (action) => {
  if (!action) return "";

  switch (action) {
    case "ALLOW":
      return "action-allow";
    case "WARNING":
      return "action-warning";
    case "CENSOR":
      return "action-censor";
    case "BLOCK":
      return "action-block";
    case "ANALYZING":
      return "action-analyzing";
    case "ERROR":
      return "action-error";
    default:
      return "action-ready";
  }
};

  return (
    <div className="app">
      <header className="header">
        <div className="logo-container">
          <div className="logo">🛡️</div>

          <div>
            <h1>GuardianNLP</h1>
            <p>Three-class multilingual moderation</p>
          </div>
        </div>

        <div className={`status ${backendReady ? "status-online" : "status-offline"}`}>
          <span className="status-dot"></span>
          {backendReady ? "Backend Ready" : "Model Unavailable"}
        </div>
      </header>

      <main className="main-container">
        <section className="hero">
          <h2>Protect conversations with intelligent moderation</h2>

          <p>
            GuardianNLP uses the IndicGuard Lite and MuRIL neural ensemble to classify text.
          </p>
        </section>

        <section className="moderation-card">
          <div className="section-header">
            <h3>Content Moderation</h3>
            <span className="model-badge">IndicGuard Lite + MuRIL</span>
          </div>

          <label htmlFor="moderation-text">
            Enter text to moderate
          </label>

          <textarea
            id="moderation-text"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Type your message here..."
            maxLength={5000}
          />

          <div className="input-footer">
            <span>{text.length} / 5000 characters</span>

            <button onClick={handleModerate} disabled={busy}>
              {busy ? "Analyzing..." : "Moderate Text"}
            </button>
          </div>
        </section>

        {result && (
          <section className="result-card" aria-live="polite">
            <div className="section-header">
              <h3>Moderation Result</h3>

              <span className={`action-badge ${getActionClass(result.action)}`}>
                {result.action}
              </span>
            </div>

            <p className="result-message">
              {result.message}
            </p>

            {result.predicted_label && (
              <>
                <div className="metrics-grid">
                  <div className="metric"><span>Predicted Label</span><strong>{result.predicted_label}</strong></div>
                  <div className="metric"><span>Confidence</span><strong>{confidencePercent(result.confidence)}</strong></div>
                  <div className="metric"><span>Final Score</span><strong>{score(result.final_score)}</strong></div>
                </div>
                {result.label_was_rule_assisted && (
                  <p className="result-message" role="status">
                    Rule-assisted effective label; confidence, score, probabilities, and Ensemble remain original model outputs.
                  </p>
                )}
                <div className="probability-list" aria-label="Class probabilities">
                  {CLASSES.map((label, index) => { const probability = result.ensemble_probabilities[index]; return <div className="probability-row" key={label}><span>{label.replace("_", " ")}</span><div className="probability-track"><div className="probability-fill" style={{ width: `${Math.max(0, Math.min(1, probability)) * 100}%` }} /></div><strong>{score(probability)}</strong></div>; })}
                </div>
              </>
            )}

            {result.predicted_label && (
              <div className="metrics-grid model-predictions" aria-label="Neural model predictions">
                <div className="metric"><span>Lite</span><strong>{result.lite_prediction || "--"}</strong></div>
                <div className="metric"><span>MuRIL</span><strong>{result.muril_prediction || "--"}</strong></div>
                <div className="metric"><span>Ensemble</span><strong>{result.ensemble_prediction || result.predicted_label}</strong></div>
              </div>
            )}

            <div className="matched-terms">
              <h4>Matched Terms</h4>

              {result.matched_terms?.length > 0 ? (
                <div className="term-list">
                  {result.matched_terms.map((term, index) => (
                    <span className="term" key={index}>
                      {term}
                    </span>
                  ))}
                </div>
              ) : (
                <span className="no-terms">Not provided by trained model</span>
              )}
            </div>
          </section>
        )}

        <section className="policy-card">
          <h3>Moderation Policies</h3>

          <div className="policy-grid">
            <div>
              <span className="policy-name allow">ALLOW</span>
              <span>Score &lt; 0.40</span>
            </div>

            <div>
              <span className="policy-name warning">WARNING</span>
              <span>0.40 – &lt; 0.70</span>
            </div>

            <div>
              <span className="policy-name censor">CENSOR</span>
              <span>0.70 – &lt; 0.85</span>
            </div>

            <div>
              <span className="policy-name block">BLOCK</span>
              <span>Score ≥ 0.85</span>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        GuardianNLP • IndicGuard Lite + MuRIL
      </footer>
    </div>
  );
}

export default App;