import { useState } from "react";
import { ask } from "../api.js";

const EXAMPLES = {
  ticket: ["What should we do first?", "Why are tickets missing their SLA?", "Which team will be overloaded next week?", "How good is the risk model?"],
  order: ["What should we do first?", "Why are orders late?", "Which region will be busy next week?", "How good is the risk model?"],
};

// "Ask your data": answers only from this analysis's facts, with the fact IDs it used.
export default function Ask({ analysisKey, live, words }) {
  const [question, setQuestion] = useState("");
  const [answers, setAnswers] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function submit(q) {
    if (q.trim().length < 3) return;
    setBusy(true);
    setError(null);
    try {
      const a = await ask(analysisKey, q);
      setAnswers((all) => [{ q, ...a }, ...all]);
      setQuestion("");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  if (!live) return <p className="banner">Ask your data needs the API. Start it and reload this page.</p>;

  return (
    <section className="ask">
      <form className="panel ask-form" onSubmit={(e) => { e.preventDefault(); submit(question); }}>
        <label htmlFor="q"><h2>Ask a question about this data</h2></label>
        <div className="ask-row">
          <input id="q" className="input" value={question} maxLength={500} onChange={(e) => setQuestion(e.target.value)}
                 placeholder={EXAMPLES[words.item][1]} />
          <button className="btn primary" disabled={busy || question.trim().length < 3}>{busy ? "Thinking…" : "Ask"}</button>
        </div>
        <div className="mini-tabs">
          {EXAMPLES[words.item].map((q) => (
            <button type="button" key={q} className="chip" onClick={() => submit(q)} disabled={busy}>{q}</button>
          ))}
        </div>
        <p className="note">Answers use only the facts in this analysis. Every claim shows the fact it came from; if nothing matches, it says so.</p>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {answers.map((a, i) => (
        <article key={i} className="panel answer">
          <p className="muted small">{a.q}</p>
          <p>{a.answer}</p>
          <p className="small">
            <span className={a.written_by === "template" ? "badge" : "badge ai"}>
              {a.written_by === "template" ? "Facts shown directly" : `Written by ${a.written_by}, numbers checked`}
            </span>
            {a.cited_ids.map((id) => <span key={id} className="fact-id">{id}</span>)}
          </p>
          {a.sources?.length > 0 && (
            <details>
              <summary>Facts used ({a.sources.length})</summary>
              <ul className="plain">{a.sources.map((s) => <li key={s.id}><strong>{s.id}</strong> {s.text}</li>)}</ul>
            </details>
          )}
        </article>
      ))}
    </section>
  );
}