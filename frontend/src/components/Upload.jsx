import { useEffect, useState } from "react";
import { connectDatabase, listUploads, runUpload, uploadFiles } from "../api.js";
import { num } from "../format.js";

const PRIORITIES = ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"];
// A starting guess from the word in the label ("P1 - Urgent" -> 4 h); the analyst can change every value.
const guessHours = (p) => {
  const w = p.toLowerCase();
  if (/critical|urgent|p1|sev ?1/.test(w)) return 4;
  if (/high|p2|sev ?2/.test(w)) return 8;
  if (/low|p4|sev ?4/.test(w)) return 72;
  return 24;
};
const hoursFor = (list) => Object.fromEntries(list.map((p) => [p, guessHours(p)]));
const ALL_TICKETS = "All tickets";
const ALL_ORDERS = "All orders";
// Days promised, guessed from the ship-mode word ("Same Day" -> 0); the analyst can change every value.
const guessDays = (m) => {
  const w = m.toLowerCase();
  if (/same/.test(w)) return 0;
  if (/express|first|next|priority|fast/.test(w)) return 2;
  if (/second/.test(w)) return 3;
  return 5;
};
const KIND = { incidents: "Support tickets", sales_orders: "Sales orders" };
const GETS = [["Fix first", "The issues worth your time, ranked."], ["At risk", "Items likely to fail, with the reason."],
              ["Forecast", "Where the numbers are heading."], ["Actions", "Recommendations you can approve."]];

// Step 1: send files or a database table. Step 2: check what we recognised and fix columns. Step 3: run.
export default function Upload({ onAnalysed }) {
  const [files, setFiles] = useState([]);
  const [company, setCompany] = useState("");
  const [db, setDb] = useState({ url: "", table: "" });
  const [result, setResult] = useState(null);
  const [overrides, setOverrides] = useState({});
  const [hours, setHours] = useState({});
  const [days, setDays] = useState({});
  const [dateOrder, setDateOrder] = useState("day");
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [earlier, setEarlier] = useState([]);
  const [source, setSource] = useState("files");      // "files" or "db"
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    listUploads().then((all) => setEarlier(all.filter((u) => u.status === "analysed"))).catch(() => {});   // no API: no list
  }, []);

  async function send(how) {
    setBusy(how);
    setError(null);
    setResult(null);
    setOverrides({});
    try {
      const r = how === "files" ? await uploadFiles(files, company) : await connectDatabase(db.url, db.table, company);
      setResult(r);
      setHours(r.priority_values?.length ? hoursFor(r.priority_values) : r.mapping?.priority ? hoursFor(PRIORITIES) : { [ALL_TICKETS]: 24 });
      const modes = r.promise_by?.values || [];
      setDays(modes.length ? Object.fromEntries(modes.map((m) => [m, guessDays(m)])) : { [ALL_ORDERS]: 4 });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  const mapped = (standard) => (standard in overrides ? overrides[standard] : result.mapping?.[standard]);
  const needsSla = result?.domain === "incidents" && !mapped("made_sla");
  const needsPromise = result?.domain === "sales_orders" && !mapped("order_estimated_delivery_date") && !mapped("promised_days");

  async function run() {
    setBusy("run");
    setError(null);
    try {
      const sla = needsSla ? { hours, default_hours: hours[ALL_TICKETS] ?? 24 } : null;
      const promise = needsPromise
        ? { column: result.promise_by?.column || null, days, default_days: days[ALL_ORDERS] ?? 4 } : null;
      const out = await runUpload(result.upload_id, overrides, sla, result.date_order === "unclear" ? dateOrder : null, promise);
      onAnalysed(out.analysis_id, company || "Your company");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  const rows = (result?.suggestions || []).filter((s) => s.required || s.needs_review || s.uploaded_column);
  const canRun = result && result.domain && result.status !== "refused";

  const step = busy === "run" ? 3 : result ? 2 : 1;
  const drop = (e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = [...e.dataTransfer.files].filter((f) => /\.(csv|xlsx)$/i.test(f.name));
    if (dropped.length) setFiles(dropped);
  };

  return (
    <section className="upload">
      <header className="upload-hero">
        <h1>Find what to fix first in your operations</h1>
        <p>Upload your support tickets or sales orders. OpsInsight cleans the data, spots what is going wrong, and ranks the actions that matter.</p>
        <ol className="steps" aria-label="Steps">
          {["Send your data", "Check columns", "Analyse"].map((label, i) => (
            <li key={label} className={i + 1 === step ? "on" : i + 1 < step ? "done" : ""} aria-current={i + 1 === step ? "step" : undefined}>
              <span className="step-no">{i + 1 < step ? "✓" : i + 1}</span>{label}
            </li>
          ))}
        </ol>
      </header>

      <div className="upload-grid">
        <div className="panel send">
          <h2>Send your data</h2>
          <p className="muted small">Support tickets or sales orders, as CSV or Excel. For sales, the orders file is enough.</p>
          <div className="source-tabs" role="tablist" aria-label="How to send your data">
            <button role="tab" aria-selected={source === "files"} className={source === "files" ? "on" : ""} onClick={() => setSource("files")}>Upload files</button>
            <button role="tab" aria-selected={source === "db"} className={source === "db" ? "on" : ""} onClick={() => setSource("db")}>Connect a database</button>
          </div>

          {source === "files" ? (
            <>
              <label htmlFor="files" className={dragging ? "dropzone over" : "dropzone"}
                     onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={drop}>
                <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M12 16V4m0 0-5 5m5-5 5 5M4 20h16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
                <strong>Drop your files here, or click to browse</strong>
                <span className="muted small">.csv or .xlsx</span>
                <input id="files" type="file" multiple accept=".csv,.xlsx" className="visually-hidden" onChange={(e) => setFiles([...e.target.files])} />
              </label>
              <label htmlFor="company" className="field-label">Company name</label>
              <input id="company" className="input" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="QuickHelp" />
              <div className="send-row">
                <span className="muted small">{files.length ? files.map((f) => f.name).join(", ") : "No files selected"}</span>
                <button className="btn primary" disabled={!files.length || busy} onClick={() => send("files")}>
                  {busy === "files" ? "Uploading…" : `Upload ${files.length > 1 ? files.length + " files" : "files"}`}
                </button>
              </div>
            </>
          ) : (
            <>
              <label htmlFor="dburl" className="field-label">Database address</label>
              <input id="dburl" className="input" value={db.url} onChange={(e) => setDb({ ...db, url: e.target.value })}
                     placeholder="postgresql://user:password@host:5432/dbname" />
              <label htmlFor="dbtable" className="field-label">Table</label>
              <input id="dbtable" className="input" value={db.table} onChange={(e) => setDb({ ...db, table: e.target.value })} placeholder="tickets" />
              <label htmlFor="company-db" className="field-label">Company name</label>
              <input id="company-db" className="input" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="QuickHelp" />
              <div className="send-row">
                <span className="muted small">Read only: OpsInsight copies the table and never writes to your database.</span>
                <button className="btn primary" disabled={!db.url || !db.table || busy} onClick={() => send("db")}>
                  {busy === "db" ? "Reading the table…" : "Read the table"}
                </button>
              </div>
            </>
          )}
        </div>

        <aside className="upload-side">
          <div className="panel">
            <h2>What we can read</h2>
            <ul className="kinds">
              <li>
                <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M5 5h14v10H9l-4 4z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" /></svg>
                <div><strong>Support tickets</strong><span className="muted small">Finds SLA breaches and what drives them.</span></div>
              </li>
              <li>
                <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M5 8h14l-1 12H6zM9 8V6a3 3 0 0 1 6 0v2" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" /></svg>
                <div><strong>Sales orders</strong><span className="muted small">Finds late deliveries and where they start.</span></div>
              </li>
            </ul>
          </div>
          <div className="panel">
            <h2>Your earlier uploads</h2>
            {earlier.length === 0 ? (
              <p className="muted small">Nothing yet. Each file you analyse appears here so you can reopen it.</p>
            ) : (
              <ul className="earlier">
                {earlier.slice(0, 5).map((u) => (
                  <li key={u.id}>
                    <div>
                      <strong>{u.company}</strong>
                      <span className="muted small">{u.main_file} · {KIND[u.domain] || u.domain} · {u.created_at.slice(0, 10)}</span>
                    </div>
                    <button className="btn" onClick={() => onAnalysed(u.id, u.company)}>Open</button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      {result && (
        <div className="panel">
          <h2>2. Check what we found</h2>
          <p className={result.status === "refused" ? "error" : ""}>{result.message}</p>
          {result.mapping_remembered && <p className="muted small">Columns filled in from your last upload for {company}. Check them below.</p>}
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
              <p className="muted small">Your file has no "SLA met" column, so we work it out: hours allowed to resolve a ticket, for each priority in your file. Any other priority gets 24 hours.</p>
              <div className="sla-grid">
                {Object.keys(hours).map((p) => (
                  <label key={p} className="small">{p}
                    <input type="number" min="1" className="input" value={hours[p]}
                           onChange={(e) => setHours({ ...hours, [p]: Number(e.target.value) })} />
                  </label>
                ))}
              </div>
            </>
          )}
          {canRun && needsPromise && (
            <>
              <h3>Days you promise for delivery</h3>
              <p className="muted small">
                Your file has no promised delivery date, so we work it out: order date + the days below
                {result.promise_by?.column ? `, for each "${result.promise_by.column}"` : ""}. An order counts as late when it arrives after that.
              </p>
              <div className="sla-grid">
                {Object.keys(days).map((m) => (
                  <label key={m} className="small">{m}
                    <input type="number" min="0" className="input" value={days[m]}
                           onChange={(e) => setDays({ ...days, [m]: Number(e.target.value) })} />
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

      <ul className="gets" aria-label="What you get">
        {GETS.map(([title, text]) => <li key={title}><strong>{title}</strong><span className="muted small">{text}</span></li>)}
      </ul>
    </section>
  );
}
