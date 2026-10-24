import { useEffect, useState } from "react";
import { getOutcomes } from "../api.js";
import { pct } from "../format.js";

const VERDICT = { worked: "good", "partly worked": "warning", "did not work": "critical" };

const fmt = (v, metric) => (metric.includes("days") ? `${v.toFixed(1)} days` : pct(v, 1));

// "Did it work?" for every approved or edited action.
export default function Outcomes({ analysisKey, live }) {
  const [state, setState] = useState({ loading: true, error: null, data: null });

  useEffect(() => {
    if (!live) return;
    getOutcomes(analysisKey)
      .then((data) => setState({ loading: false, error: null, data }))
      .catch((e) => setState({ loading: false, error: e.message, data: null }));
  }, [analysisKey, live]);

  if (!live) return <p className="banner">Results need the API: approvals are saved there. Start it, approve an action, then come back.</p>;
  if (state.loading) return <p className="muted pad">Checking results…</p>;
  if (state.error) return <p className="error pad">{state.error}</p>;
  const { actions, rule, caution } = state.data;

  return (
    <section className="recs">
      <p className="muted">Compares the period before with the period after. {rule}.</p>
      {actions.length === 0 && <p className="panel">No approved actions yet. Approve one in Actions, then come back here.</p>}
      {actions.map((a) => {
        const o = a.overall;
        return (
          <article key={a.recommendation_id} className="panel outcome">
            <div className="panel-head">
              <h2>{a.title}</h2>
              <span className={`verdict v-${VERDICT[o.verdict]}`}>{o.verdict}</span>
            </div>
            <p className="impact-line tnum">
              <span>{pct(o.before, 1)}</span><span className="arrow" aria-hidden="true">›</span>
              <span>{pct(o.after, 1)}</span>
              <span className="muted small">hoped for {pct(o.expected, 1)}</span>
            </p>
            <p className="muted small">{o.metric.replaceAll("_", " ")}, before and after. Approved {a.approved_at.slice(0, 10)}.</p>
            {a.own_target && (
              <p className="small">
                Its own target, {a.own_target.metric.replaceAll("_", " ")}: {fmt(a.own_target.before, a.own_target.metric)} → {fmt(a.own_target.after, a.own_target.metric)}{" "}
                {a.own_target.moved_the_right_way ? "(moved the right way)" : "(did not move)"}.
              </p>
            )}
            {a.note && <p className="small warn-text">{a.note}</p>}
          </article>
        );
      })}
      <p className="note">{caution}</p>
    </section>
  );
}