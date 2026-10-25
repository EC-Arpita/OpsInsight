import { useState } from "react";
import { excelUrl, sendDecision } from "../api.js";
import { num, pct } from "../format.js";

const savedName = () => { try { return localStorage.getItem("opsinsight_name") || ""; } catch { return ""; } };
const EDITABLE = [["title", "Title"], ["description", "What to do"], ["suggested_owner", "Owner"]];

// Actions the manager approves, rejects or edits. Decisions go to the API (or stay on screen in sample mode).
export default function Recommendations({ analysisKey, recs, impact, words, live, onDecided }) {
  const [items, setItems] = useState(recs);
  const [busy, setBusy] = useState(null);
  const [message, setMessage] = useState(null);
  const [comments, setComments] = useState({});
  const [editing, setEditing] = useState(null);       // { id, title, description, suggested_owner }
  const [name, setName] = useState(savedName);
  const decidedBy = name.trim() || "Ops manager";
  const avoidedKey = Object.keys(recs[0]?.expected_impact || {}).find((k) => k.endsWith("_avoided_per_period"));
  const rangeKey = Object.keys(recs[0]?.expected_impact || {}).find((k) => k.startsWith("range_"));
  const problems = words.item === "ticket" ? "breaches" : "late orders";

  async function decide(rec, decision, edited) {
    setBusy(rec.id);
    setMessage(null);
    const comment = comments[rec.id] || "";
    try {
      const approval = live
        ? (await sendDecision(analysisKey, rec.id, decision, decidedBy, comment, edited)).approval
        : { decision, decided_by: decidedBy, comment, edited, decided_at: new Date().toISOString() };
      const updated = { ...rec, ...(edited || {}), status: decision, approval };
      setItems((all) => all.map((r) => (r.id === rec.id ? updated : r)));
      onDecided?.(updated);
      setEditing(null);
      setMessage(live ? `${rec.id} ${decision}.` : `${rec.id} ${decision} on this screen only. Start the API to save decisions.`);
    } catch (e) {
      setMessage(e.message);
    } finally {
      setBusy(null);
    }
  }

  function saveEdit(rec) {
    const changed = Object.fromEntries(EDITABLE.map(([k]) => [k, editing[k]]).filter(([k, v]) => v !== rec[k]));
    if (!Object.keys(changed).length) return setEditing(null);
    decide(rec, "edited", changed);
  }

  return (
    <section className="recs">
      <div className="recs-bar no-print">
        <label className="small" htmlFor="decider">Your name</label>
        <input id="decider" className="input" value={name} placeholder="Ops manager"
               onChange={(e) => { setName(e.target.value); try { localStorage.setItem("opsinsight_name", e.target.value); } catch {} }} />
        {live && <a className="btn" href={excelUrl(analysisKey)} download>Download Excel</a>}
        <button className="btn" onClick={() => window.print()}>Print / Save as PDF</button>
      </div>
      <p className="muted">
        Baseline this period: {pct(impact.baseline[words.rateKey], 1)} {words.problem} rate. Each estimate is separate; they overlap, so don't add them up.
      </p>
      {message && <p className="toast" role="status">{message}</p>}
      {items.map((r) => {
        const ei = r.expected_impact;
        const isEditing = editing?.id === r.id;
        return (
          <article key={r.id} className={`rec rec-${r.status}`}>
            <div className="rec-main">
              <p className="rec-tags">
                <span className="fact-id">{r.id}</span>
                {r.quick_win && <span className="badge ai">Quick win</span>}
              </p>
              {isEditing ? (
                <div className="form-grid">
                  {EDITABLE.map(([k, label]) => (
                    <label key={k} className="small">{label}
                      {k === "description"
                        ? <textarea className="input" rows="3" value={editing[k]} onChange={(e) => setEditing({ ...editing, [k]: e.target.value })} />
                        : <input className="input" value={editing[k]} onChange={(e) => setEditing({ ...editing, [k]: e.target.value })} />}
                    </label>
                  ))}
                  <p className="muted small">The expected impact can't be edited: it is calculated, not chosen.</p>
                </div>
              ) : (
                <>
                  <h2>{r.title}</h2>
                  <p>{r.description}</p>
                </>
              )}
              <details>
                <summary>How this was estimated</summary>
                <p>{r.estimate_method}</p>
                <p className="muted">Assumes: {r.assumptions.join("; ")}.</p>
                <p className="muted">Based on: {r.evidence.join(" ")}</p>
              </details>
              <dl className="rec-meta">
                <div><dt>Confidence</dt><dd>{r.confidence}</dd></div>
                <div><dt>Effort</dt><dd>{r.effort}</dd></div>
                <div><dt>Owner</dt><dd>{r.suggested_owner}</dd></div>
              </dl>
            </div>
            <div className="rec-impact">
              <p className="impact-line tnum">
                <span>{pct(ei.baseline, 1)}</span>
                <span className="arrow" aria-hidden="true">›</span>
                <span className="impact-to">{pct(ei.expected, 1)}</span>
              </p>
              <p className="muted small">
                {words.problem} rate; about {num(ei[avoidedKey])} fewer {problems} per period
                {ei[rangeKey] ? ` (range ${num(ei[rangeKey][0])} to ${num(ei[rangeKey][1])})` : ""}
              </p>
              {isEditing ? (
                <div className="rec-buttons">
                  <button className="btn primary" disabled={busy === r.id} onClick={() => saveEdit(r)}>Save and approve</button>
                  <button className="btn" onClick={() => setEditing(null)}>Cancel</button>
                </div>
              ) : r.status === "pending" ? (
                <>
                  <label className="small" htmlFor={`c-${r.id}`}>Comment (optional)</label>
                  <input id={`c-${r.id}`} className="input" value={comments[r.id] || ""}
                         onChange={(e) => setComments({ ...comments, [r.id]: e.target.value })} />
                  <div className="rec-buttons">
                    <button className="btn primary" disabled={busy === r.id} onClick={() => decide(r, "approved")}>Approve</button>
                    <button className="btn" disabled={busy === r.id} onClick={() => decide(r, "rejected")}>Reject</button>
                    <button className="btn" disabled={busy === r.id}
                            onClick={() => setEditing({ id: r.id, title: r.title, description: r.description, suggested_owner: r.suggested_owner })}>
                      Edit
                    </button>
                  </div>
                </>
              ) : (
                <p className={`decided d-${r.status}`}>
                  {r.status === "approved" ? "Approved" : r.status === "rejected" ? "Rejected" : "Edited and approved"} by {r.approval?.decided_by}
                  {r.approval?.comment ? `: “${r.approval.comment}”` : ""}
                </p>
              )}
            </div>
          </article>
        );
      })}
    </section>
  );
}
