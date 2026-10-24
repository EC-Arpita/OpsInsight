import { pct } from "../format.js";

// "Why is it happening?" Each cause shows the comparison that proves it.
export default function RootCauses({ causes }) {
  return (
    <section className="panel">
      <div className="panel-head"><h2>Why it's happening</h2></div>
      <ol className="causes">
        {causes.map((c) => {
          const rates = c.evidence.filter((e) => e.breach_rate !== undefined || e.late_rate !== undefined);
          const max = Math.max(...rates.map((e) => e.breach_rate ?? e.late_rate), 0.01);
          return (
            <li key={c.id} className="cause">
              <div className="cause-top">
                <h3>{c.title}</h3>
                <span className={`conf conf-${c.confidence}`}>{c.confidence} confidence</span>
              </div>
              <p>{c.finding}</p>
              {rates.length > 0 && (
                <div className="evidence">
                  {rates.map((e) => {
                    const v = e.breach_rate ?? e.late_rate;
                    return (
                      <div key={e.segment} className="ev-row">
                        <span className="ev-label">{e.segment}</span>
                        <span className="ev-bar"><span style={{ width: `${(v / max) * 100}%` }} /></span>
                        <span className="tnum ev-val">{pct(v)}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}