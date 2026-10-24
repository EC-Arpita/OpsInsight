import { num } from "../format.js";

// "What to fix first": the ranked findings, the work-first queue and next week's plan.
export default function FixFirst({ data, words }) {
  const f = data.ranked_findings;
  if (!f) return <p className="muted pad">This analysis has no ranked findings yet. Run the engine refresh.</p>;
  const top = Math.max(...f.items.map((c) => c.score || 0), 0.01);
  const plan = data.workload_plan;
  const queue = Array.isArray(data.priority_queue) ? data.priority_queue : data.priority_queue?.today || [];
  const replays = data.impact_simulation?.replays?.queues?.teams || [];

  return (
    <div className="fix">
      <section className="panel">
        <div className="panel-head"><h2>Fix this first</h2><span className="muted small">{f.rule}</span></div>
        <ol className="ranked">
          {f.items.map((c) => (
            <li key={c.id} className="ranked-item">
              <span className="rank tnum" aria-hidden="true">{c.rank}</span>
              <div>
                <h3>{c.title}</h3>
                <p>{c.finding}</p>
                <p className="why-line">{c.why}</p>
                {c.related && <p className="muted small">Also explains the forecast {c.related}.</p>}
              </div>
              <span className="score-bar" title={`score ${c.score ?? "not measured"}`}>
                <span style={{ width: `${((c.score || 0) / top) * 100}%` }} />
              </span>
            </li>
          ))}
        </ol>
        {f.context.length > 0 && (
          <p className="note">Context, not a cause to fix: {f.context.map((c) => c.finding).join(" ")}</p>
        )}
        {(f.watch.length > 0 || f.history.length > 0) && (
          <details>
            <summary>Watch list and past weeks ({f.watch.length + f.history.length})</summary>
            <ul className="plain">
              {[...f.watch, ...f.history].map((c) => <li key={c.id}><strong>{c.title}.</strong> <span className="muted">{c.why}</span></li>)}
            </ul>
          </details>
        )}
        <p className="note">{f.note}</p>
      </section>

      {plan && (plan.moves.length > 0 || plan.staffing_gaps.length > 0) && (
        <section className="panel">
          <div className="panel-head"><h2>Next week's plan</h2>
            <span className="muted small">Only low-risk tickets move, never more than one day.</span></div>
          <div className="two-col tight">
            <div>
              <h3>Move work</h3>
              <table className="table">
                <thead><tr><th scope="col">Day</th><th scope="col" className="r">Tickets</th><th scope="col">From → to</th></tr></thead>
                <tbody>
                  {[...plan.moves].sort((a, b) => a.date.localeCompare(b.date)).map((m, i) => (
                    <tr key={i} title={m.why}>
                      <td>{m.date}</td><td className="r tnum">{m.tickets}</td>
                      <td>{m.from_team === m.to_team ? `${m.from_team}: to ${m.to_date}` : `${m.from_team} → ${m.to_team}`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div>
              <h3>Add cover</h3>
              <table className="table">
                <thead><tr><th scope="col">Day</th><th scope="col">Team</th><th scope="col" className="r">Extra tickets</th></tr></thead>
                <tbody>
                  {plan.staffing_gaps.map((g, i) => (
                    <tr key={i} className="flagged"><td>{g.date}</td><td>{g.team}</td><td className="r tnum">{g.tickets}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          {replays.filter((t) => t.baseline.ticket_nights_waiting > 1).map((t) => (
            <p key={t.team} className="small">
              <strong>{t.team}:</strong> tickets left waiting overnight next week fall from{" "}
              <span className="tnum">{num(t.baseline.ticket_nights_waiting)}</span> to{" "}
              <span className="tnum">{num(t.with_plan.ticket_nights_waiting)}</span> with the moves and the cover.
            </p>
          ))}
          <ul className="checks">
            {Object.entries(plan.check).map(([k, ok]) => (
              <li key={k} className={ok ? "ok" : "bad"}>{ok ? "✓" : "✗"} {k.replaceAll("_", " ")}</li>
            ))}
          </ul>
        </section>
      )}

      {queue.length > 0 && (
        <section className="panel">
          <div className="panel-head"><h2>Work on these {words.items} first</h2>
            <span className="muted small">{words.item === "ticket" ? "risk × priority weight" : "money at risk = risk × order value"}</span></div>
          <table className="table">
            <thead><tr><th scope="col">#</th><th scope="col">{words.item === "ticket" ? "Ticket" : "Order"}</th>
              <th scope="col">{words.item === "ticket" ? "Team" : "Customer state"}</th><th scope="col">Why first</th></tr></thead>
            <tbody>
              {queue.slice(0, 8).map((q) => (
                <tr key={q.id}><td className="tnum">{q.rank}</td><td>{words.item === "order" ? q.id.slice(0, 10) + "…" : q.id}</td>
                  <td>{q.team || q.region}</td><td className="muted">{q.why}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}