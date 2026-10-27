import { useState } from "react";
import WhyPanel from "./WhyPanel.jsx";

// Left: items ranked by risk. Right: why the selected one is risky.
export default function RiskList({ predictions, pickup, words }) {
  // Stage 1 = scored when the order is approved; stage 2 = scored again once the courier has it (sales only)
  const [stage, setStage] = useState("approval");
  const section = stage === "pickup" && pickup ? pickup : predictions;
  const items = section.items;
  const [selected, setSelected] = useState(items[0]?.id);
  function pick(next) {
    setStage(next);
    setSelected((next === "pickup" && pickup ? pickup : predictions).items[0]?.id);
  }
  const current = items.find((i) => i.id === selected);

  if (!items.length) return <p className="muted pad">No {words.items} at risk in this period.</p>;

  return (
    <div className="risk-layout">
      <section className="panel risk-list" aria-label={`${words.items} at risk`}>
        <div className="panel-head">
          <h2>Highest-risk {words.items}</h2>
          {pickup && (
            <div className="mini-tabs" role="group" aria-label="When the risk is scored">
              <button className={stage === "approval" ? "chip on" : "chip"} aria-pressed={stage === "approval"}
                      onClick={() => pick("approval")}>At approval</button>
              <button className={stage === "pickup" ? "chip on" : "chip"} aria-pressed={stage === "pickup"}
                      onClick={() => pick("pickup")}>At courier pickup</button>
            </div>
          )}
        </div>
        <ul>
          {items.map((it) => {
            const c = it.context;
            const route = [c.seller_state, c.customer_state].filter(Boolean).join(" to ");   // orders-only files have no seller or category
            const sub = words.item === "ticket" ? `${c.team}, ${c.category}` : [route, c.category?.replaceAll("_", " ")].filter(Boolean).join(", ");
            return (
              <li key={it.id}>
                <button
                  className={it.id === selected ? "risk-row on" : "risk-row"}
                  onClick={() => setSelected(it.id)}
                  aria-pressed={it.id === selected}
                >
                  <span className="risk-id">{words.item === "order" ? it.id.slice(0, 10) + "…" : it.id}</span>
                  <span className="risk-sub">{sub}</span>
                  <span className={`risk-score lvl-${it.prediction.risk_level}`}>{Math.round(it.prediction.risk * 100)}%</span>
                </button>
              </li>
            );
          })}
        </ul>
        <p className="note">
          High = top 10% of risk in the current period.
          {stage === "pickup" && pickup
            ? ` Scored again once the courier has the order (AUC ${pickup.model_performance.test_auc}, ${pickup.model_performance.lift_top_10pct}× more late orders in the top 10%).`
            : ""}
        </p>
      </section>
      {current && <WhyPanel item={current} words={words} />}
    </div>
  );
}