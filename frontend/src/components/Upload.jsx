import { useState } from "react";
import { connectDatabase, runUpload, uploadFiles } from "../api.js";
import { num } from "../format.js";

const PRIORITIES = ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"];
const DEFAULT_HOURS = { "1 - Critical": 4, "2 - High": 8, "3 - Moderate": 24, "4 - Low": 72 };
const KIND = { incidents: "Support tickets", sales_orders: "Sales orders" };

// Step 1: send files or a database table. Step 2: check what we recognised and fix columns. Step 3: run.
export default function Upload({ onAnalysed }) {
  const [files, setFiles] = useState([]);
  const [company, setCompany] = useState("");
  const [db, setDb] = useState({ url: "", table: "" });
  const [result, setResult] = useState(null);
  const [overrides, setOverrides] = useState({});
  const [hours, setHours] = useState(DEFAULT_HOURS);
  const [dateOrder, setDateOrder] = useState("day");
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);

  async function send(how) {
    setBusy(how);
    setError(null);
    setResult(null);
    setOverrides({});
    try {
      setResult(how === "files" ? await uploadFiles(files, company) : await connectDatabase(db.url, db.table, company));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  const mapped = (standard) => (standard in overrides ? overrides[standard] : result.mapping?.[standard]);
  const needsSla = result?.domain === "incidents" && !mapped("made_sla");

  async function run() {
    setBusy("run");
    setError(null);
    try {
      const sla = needsSla ? { hours, default_hours: 24 } : null;
      const out = await runUpload(result.upload_id, overrides, sla, result.date_order === "unclear" ? dateOrder : null);
      onAnalysed(out.analysis_id, company || "Your company");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  const rows = (result?.suggestions || []).filter((s) => s.required || s.needs_review || s.uploaded_column);
  const canRun = result && result.domain && result.status !== "refused";

  return (
    <section className="upload">
      <div className="panel">
        <h2>1. Send your data</h2>
        <p className="muted small">Support tickets or sales orders, as CSV or Excel. For sales, the orders file is enough.</p>
        <div className="form-grid">
          <label htmlFor="company">Company name</label>
          <input id="company" className="input" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="QuickHelp" />
          <label htmlFor="files">Files</label>
          <input id="files" type="file" multiple accept=".csv,.xlsx" onChange={(e) => setFiles([...e.target.files])} />
        </div>
        <button className="btn primary" disabled={!files.length || busy} onClick={() => send("files")}>
          {busy === "files" ? "Uploading…" : `Upload ${files.length || ""} file${files.length === 1 ? "" : "s"}`}
        </button>
        <details>
          <summary>Or connect a database</summary>
          <div className="form-grid">
            <label htmlFor="dburl">Database address</label>
            <input id="dburl" className="input" value={db.url} onChange={(e) => setDb({ ...db, url: e.target.value })}
                   placeholder="postgresql://user:password@host:5432/dbname" />
            <label htmlFor="dbtable">Table</label>
            <input id="dbtable" className="input" value={db.table} onChange={(e) => setDb({ ...db, table: e.target.value })} placeholder="tickets" />
          </div>
          <button className="btn" disabled={!db.url || !db.table || busy} onClick={() => send("db")}>
            {busy === "db" ? "Reading the table…" : "Read the table"}
          </button>
          <p className="note">Read only: OpsInsight copies the table and never writes to your database.</p>
        </details>
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      {result && (
        <div className="panel">
          <h2>2. Check what we found</h2>
          <p className={result.status === "refused" ? "error" : ""}>{result.message}</p>
          <dl className="facts">
            <div><dt>Main file</dt><dd>{result.main_file}</dd></div>
            <div><dt>Rows</dt><dd className="tnum">{num(result.rows)}</dd></div>
            <div><dt>Kind of data</dt><dd>{KIND[result.domain] || "Not recognised"}</dd></div>
          </dl>
          {canRun && rows.length > 0 && (
            <>
              <h3>Columns we need</h3>
              <p className="muted small">Check each match. Orange rows need your eye. Choose "Not in my file" if a column doesn't exist.</p>
              <table className="table">
                <thead><tr><th scope="col">We need</th><th scope="col">Your column</th><th scope="col">Why we matched it</th></tr></thead>
                <tbody>
                  {rows.map((s) => (
                    <tr key={s.standard} className={s.needs_review ? "flagged" : ""}>
                      <td>{s.standard}{s.required ? "" : " (optional)"}</td>
                      <td>
                        <select className="input" aria-label={`Your column for ${s.standard}`} value={mapped(s.standard) || ""}
                                onChange={(e) => setOverrides({ ...overrides, [s.standard]: e.target.value })}>
                          <option value="">Not in my file</option>
                          {result.columns.map((c) => <option key={c} value={c}>{c}</option>)}
                        </select>
                      </td>
                      <td className="muted small">{s.reason}{s.if_missing && !mapped(s.standard) ? `; if missing: ${s.if_missing}` : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
          {canRun && result.date_order === "unclear" && (
            <>
              <h3>How are your dates written?</h3>
              <p className="muted small">Dates like 05/03/2024 could mean 5 March or 3 May. Choose how your system writes them.</p>
              <div className="mini-tabs" role="group" aria-label="Date order">
                <button type="button" className={dateOrder === "day" ? "chip on" : "chip"} aria-pressed={dateOrder === "day"}
                        onClick={() => setDateOrder("day")}>Day first (31/05/2024)</button>
                <button type="button" className={dateOrder === "month" ? "chip on" : "chip"} aria-pressed={dateOrder === "month"}
                        onClick={() => setDateOrder("month")}>Month first (05/31/2024)</button>
              </div>
            </>
          )}
          {canRun && needsSla && (
            <>
              <h3>Your SLA rules</h3>
              <p className="muted small">Your file has no "SLA met" column, so we work it out: hours allowed to resolve a ticket, per priority.</p>
              <div className="sla-grid">
                {PRIORITIES.map((p) => (
                  <label key={p} className="small">{p}
                    <input type="number" min="1" className="input" value={hours[p]}
                           onChange={(e) => setHours({ ...hours, [p]: Number(e.target.value) })} />
                  </label>
                ))}
              </div>
            </>
          )}
          {canRun && (
            <div className="run-row">
              <h2>3. Analyse</h2>
              <button className="btn primary" disabled={busy} onClick={run}>{busy === "run" ? "Analysing…" : "Run the analysis"}</button>
              {busy === "run" && <p className="muted small" role="status">This takes 10 to 60 seconds: cleaning, risk model, forecast, decisions.</p>}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
