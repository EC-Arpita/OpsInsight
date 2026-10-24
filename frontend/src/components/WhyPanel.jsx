import { num, pct } from "../format.js";

const LABELS = {
  opened_at: "Opened", team: "Team", category: "Category", subcategory: "Subcategory", priority: "Priority",
  location: "Location", contact_type: "Channel", approved_at: "Approved", promised_by: "Promised by",
  customer_state: "Customer state", customer_city: "Customer city", seller_id: "Seller", seller_state: "Seller state",
  order_value: "Order value (R$)", n_items: "Items", payment_type: "Payment",
};

function show(v) {
  if (typeof v === "number") return num(v);
  if (/^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2})?$/.test(v)) {
    const d = new Date(v);
    const opts = { day: "numeric", month: "short", year: "numeric" };
    return v.includes("T")
      ? d.toLocaleString("en-GB", { ...opts, hour: "2-digit", minute: "2-digit" })
      : d.toLocaleDateString("en-GB", opts);
  }
  return String(v).replaceAll("_", " ");
}

// The signature view: each reason is a bar pushing risk up (right) or down (left).
export default function WhyPanel({ item, words }) {
  const maxEffect = Math.max(...item.top_reasons.map((r) => Math.abs(r.effect)), 0.01);
  const sim = item.similar_past_cases;
  const simRate = sim.breach_rate ?? sim.late_rate;
  return (
    <section className="panel why" aria-live="polite">
      <div className="why-head">
        <div>
          <p className="muted small">{words.item === "ticket" ? "Ticket" : "Order"} {item.id}</p>
          <h2>Why this {words.item} is at risk</h2>
        </div>
        <div className={`why-score lvl-${item.prediction.risk_level}`}>
          <span className="tnum">{Math.round(item.prediction.risk * 100)}%</span>
          <span className="small">chance of {words.problem}</span>
        </div>
      </div>

      <div className="reasons">
        {item.top_reasons.map((r) => {
          const w = (Math.abs(r.effect) / maxEffect) * 50;
          const up = r.effect > 0;
          return (
            <div key={r.feature} className="reason">
              <div className="reason-text">
                <strong>{r.label}</strong>
                <span className="muted">
                  {typeof r.value === "number" ? num(r.value) : r.value}
                  {" "}(usually {typeof r.typical_value === "number" ? num(r.typical_value) : r.typical_value})
                </span>
              </div>
              <div className="push" aria-label={`${r.direction}, strength ${Math.abs(r.effect).toFixed(2)}`}>
                <span className="axis" />
                <span
                  className={up ? "push-bar up" : "push-bar down"}
                  style={up ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }}
                />
              </div>
              <span className={up ? "dir up" : "dir down"}>{up ? "Raises risk" : "Lowers risk"}</span>
            </div>
          );
        })}
      </div>

      <div className="why-grid">
        <div>
          <h3>Details</h3>
          <dl className="facts">
            {Object.entries(item.context).map(([k, v]) => (
              <div key={k}><dt>{LABELS[k] || k}</dt><dd>{show(v)}</dd></div>
            ))}
          </dl>
        </div>
        <div>
          <h3>Similar past {words.items}</h3>
          <p>
            <span className="big tnum">{pct(simRate)}</span> of {num(sim.count)} similar {words.items} ended in {words.problem}.
          </p>
          <p className="muted small">Similar = {sim.rule}.</p>
          <h3>Do this now</h3>
          <p className="action">{item.recommended_action.action}</p>
        </div>
      </div>
    </section>
  );
}