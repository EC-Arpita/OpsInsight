import { fmtValue, STATUS_TEXT } from "../format.js";

// One strip of KPIs. Change is coloured by whether it is good or bad, never by sign.
export default function KpiStrip({ kpis }) {
  return (
    <section className="kpis" aria-label="Key numbers">
      {kpis.map((k) => {
        const arrow = k.change_pct > 0 ? "▲" : k.change_pct < 0 ? "▼" : "■";
        const tone = k.trend === "improving" ? "up-good" : k.trend === "worsening" ? "up-bad" : "neutral";
        return (
          <div key={k.id} className={`kpi status-${k.status}`} title={k.definition}>
            <p className="kpi-label">{k.label}</p>
            <p className="kpi-value">{fmtValue(k.value, k.unit)}</p>
            <p className={`kpi-change ${tone}`}>
              {k.change_pct === null ? "No earlier period" : `${arrow} ${Math.abs(k.change_pct)}% vs previous`}
            </p>
            {STATUS_TEXT[k.status] && <p className="kpi-status">{STATUS_TEXT[k.status]}</p>}
          </div>
        );
      })}
    </section>
  );
}