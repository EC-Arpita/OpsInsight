import { useEffect, useState } from "react";
import { loadAnalysis } from "./api.js";
import { DOMAINS, fmtDate } from "./format.js";
import KpiStrip from "./components/KpiStrip.jsx";
import TrendCharts from "./components/TrendCharts.jsx";
import Breakdown from "./components/Breakdown.jsx";
import RootCauses from "./components/RootCauses.jsx";
import DataHealth from "./components/DataHealth.jsx";
import RiskList from "./components/RiskList.jsx";
import Recommendations from "./components/Recommendations.jsx";
import Forecast from "./components/Forecast.jsx";
import FixFirst from "./components/FixFirst.jsx";
import Outcomes from "./components/Outcomes.jsx";
import Ask from "./components/Ask.jsx";
import Upload from "./components/Upload.jsx";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "fix", label: "Fix first" },
  { id: "risk", label: "At risk" },
  { id: "forecast", label: "Forecast" },
  { id: "actions", label: "Actions" },
  { id: "results", label: "Results" },
  { id: "ask", label: "Ask" },
];

export default function App() {
  // key = which analysis: "incidents" or "sales" (the demo data) or an upload id like "upload-3bdaa119"
  const [key, setKey] = useState("incidents");
  const [company, setCompany] = useState(null);
  const [tab, setTab] = useState("upload");
  const [state, setState] = useState({ loading: true, error: null, data: null, source: null });

  useEffect(() => {
    if (tab === "upload") return;
    setState((s) => ({ ...s, loading: true, error: null }));
    loadAnalysis(key)
      .then(({ data, source }) => setState({ loading: false, error: null, data, source }))
      .catch((e) => setState({ loading: false, error: e.message, data: null, source: null }));
  }, [key, tab === "upload"]);

  const { data, source, loading, error } = state;
  const domain = data ? (data.meta.domain === "support_tickets" ? "incidents" : "sales") : key in DOMAINS ? key : "incidents";
  const words = DOMAINS[domain];
  const live = source === "api";

  function pickDemo(id) {
    setKey(id);
    setCompany(null);
    if (tab === "upload") setTab("overview");
  }

  // keep the app's copy of a card in step with a decision, so the count and other tabs stay right
  function decided(rec) {
    setState((s) => ({ ...s, data: { ...s.data, recommendations: s.data.recommendations.map((r) => (r.id === rec.id ? rec : r)) } }));
  }

  function analysed(uploadId, name) {
    setKey(uploadId);
    setCompany(name);
    setTab("fix");
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">OpsInsight</span>
        </div>
        <div className="top-actions">
          {company && <span className="company-chip">Your data: {company}</span>}
          <div className="domain-switch" role="group" aria-label="Demo data">
            {Object.entries(DOMAINS).map(([id, d]) => (
              <button key={id} className={id === key && tab !== "upload" ? "seg on" : "seg"} aria-pressed={id === key}
                      onClick={() => pickDemo(id)}>
                {d.name}
              </button>
            ))}
          </div>
          <button className={tab === "upload" ? "btn primary" : "btn"} onClick={() => setTab("upload")}>Upload data</button>
        </div>
      </header>

      {tab === "upload" ? (
        <main><Upload onAnalysed={analysed} /></main>
      ) : (
        <>
          {source === "sample" && (
            <p className="banner">
              Showing the sample file because the API isn't running. Start it with
              <code>uvicorn opsinsight.main:app --reload</code> inside the <code>api</code> folder.
            </p>
          )}

          {data && (
            <section className="context">
              <h1>{data.meta.dataset_name}</h1>
              <p>
                Predicting <strong>{data.meta.target.definition}</strong>, {data.meta.prediction_moment}.
                Current period {fmtDate(data.meta.current_period.start)} to {fmtDate(data.meta.current_period.end)},
                compared with the {fmtDate(data.meta.previous_period.start)} to {fmtDate(data.meta.previous_period.end)} period.
              </p>
            </section>
          )}

          <nav className="tabs" aria-label="Sections">
            {TABS.map((t) => (
              <button key={t.id} className={t.id === tab ? "tab on" : "tab"} aria-current={t.id === tab ? "page" : undefined}
                      onClick={() => setTab(t.id)}>
                {t.label}
                {t.id === "risk" && data ? <span className="count">{data.predictions.count_shown}</span> : null}
                {t.id === "actions" && data ? (
                  <span className="count">{data.recommendations.filter((r) => r.status === "pending").length}</span>
                ) : null}
              </button>
            ))}
          </nav>

          <main>
            {loading && <p className="muted pad">Loading analysis…</p>}
            {error && <p className="error pad">{error}</p>}
            {data && !loading && tab === "overview" && (
              <>
                <KpiStrip kpis={data.kpis} />
                <TrendCharts trends={data.trends} words={words} anomalies={data.anomalies} />
                <div className="two-col">
                  <Breakdown breakdowns={data.breakdowns} words={words} />
                  <RootCauses causes={data.root_causes} />
                </div>
                <DataHealth quality={data.data_quality} model={data.model_performance} />
              </>
            )}
            {data && !loading && tab === "fix" && <FixFirst data={data} words={words} />}
            {data && !loading && tab === "risk" && <RiskList key={key} predictions={data.predictions} words={words} />}
            {data && !loading && tab === "forecast" && <Forecast key={key} forecast={data.forecast} />}
            {data && !loading && tab === "actions" && (
              <Recommendations key={key} analysisKey={key} recs={data.recommendations} impact={data.impact_simulation}
                               words={words} live={live} onDecided={decided} />
            )}
            {data && !loading && tab === "results" && <Outcomes key={key} analysisKey={key} live={live} />}
            {data && !loading && tab === "ask" && <Ask key={key} analysisKey={key} live={live} words={words} />}
          </main>
        </>
      )}
    </div>
  );
}