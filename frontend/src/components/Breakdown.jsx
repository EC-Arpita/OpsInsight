import { useState } from "react";
import { num, pct } from "../format.js";

// "Where is it worst?" One table per dimension, bar = problem rate.
export default function Breakdown({ breakdowns, words }) {
  const [i, setI] = useState(0);
  const b = breakdowns[i];
  const max = Math.max(...b.rows.map((r) => r[words.rateKey]));
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Where it's worst</h2>
        <div className="mini-tabs" role="group" aria-label="Break down by">
          {breakdowns.map((x, j) => (
            <button key={x.dimension} className={j === i ? "chip on" : "chip"} aria-pressed={j === i} onClick={() => setI(j)}>
              {x.label}
            </button>
          ))}
        </div>
      </div>
      <table className="table">
        <thead>
          <tr>
            <th scope="col">{b.label}</th>
            <th scope="col" className="r">{words.items}</th>
            <th scope="col">{words.problem} rate</th>
            <th scope="col" className="r">Share of all</th>
          </tr>
        </thead>
        <tbody>
          {b.rows.map((r) => (
            <tr key={r.value} className={r.flag === "above_average" ? "flagged" : ""}>
              <td className="ellipsis" title={r.value}>{r.value.length > 14 ? `${r.value.slice(0, 8)}…` : r.value}</td>
              <td className="r tnum">{num(r[words.countKey])}</td>
              <td>
                <div className="bar-cell">
                  <span className="bar" style={{ width: `${(r[words.rateKey] / max) * 100}%` }} />
                  <span className="tnum">{pct(r[words.rateKey], 1)}</span>
                </div>
              </td>
              <td className="r tnum">{pct(r[words.problemShare])}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">Highlighted rows are more than 20% worse than average. Sorted by number of {words.problem === "SLA breach" ? "breaches" : "late orders"}.</p>
    </section>
  );
}