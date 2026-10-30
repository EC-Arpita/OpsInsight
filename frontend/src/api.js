// All talking to the backend lives here, so the rest of the app never changes
// when the API changes. Set VITE_API_URL in a .env file to point somewhere else.
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// One place that calls the API and turns its error answers into a readable message.
async function call(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API}${path}`, options);
  } catch {
    throw new Error("Could not reach the server. It may be starting up: wait a minute and try again. (Running locally? Start it with: uvicorn opsinsight.main:app --reload --port 8000)");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `The API answered ${res.status}`);
  return body;
}

const json = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// Load the full analysis JSON. key = "incidents", "sales" or an upload id.
// If the API isn't running, the two demo analyses fall back to the sample files in /public.
export async function loadAnalysis(key) {
  try {
    return { data: await call(`/api/analysis/${key}`), source: "api" };
  } catch (e) {
    if (key !== "incidents" && key !== "sales") throw e;
    const res = await fetch(`/example_${key}.json`);
    if (!res.ok) throw new Error(`Couldn't load the API or the sample file example_${key}.json`);
    return { data: await res.json(), source: "sample" };
  }
}

// Save a manager's decision. edited = {title, description, suggested_owner} for an "edited" decision.
export function sendDecision(key, recId, decision, decidedBy, comment, edited) {
  return call(`/api/recommendations/${key}/${recId}/decision`,
              json({ decision, decided_by: decidedBy, comment: comment || null, edited: edited || null }));
}

// Upload files (a FileList or array of File) and get back what was recognised.
export function uploadFiles(files, company) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  form.append("company", company || "unknown");
  return call("/api/uploads", { method: "POST", body: form });
}

export const connectDatabase = (url, table, company) => call("/api/uploads/connect", json({ url, table, company: company || "unknown" }));
// The analysis runs in the background (a big file takes about a minute, longer than some hosts keep a request
// open); we then ask every few seconds whether it has finished.
export async function runUpload(id, mapping, slaRules, dateOrder, promiseRules) {
  await call(`/api/uploads/${id}/run?background=true`, json({ mapping, sla_rules: slaRules || null,
                                                             date_order: dateOrder || null, promise_rules: promiseRules || null }));
  for (let i = 0; i < 200; i++) {                       // up to 10 minutes
    await new Promise((r) => setTimeout(r, 3000));
    let up;
    try { up = await call(`/api/uploads/${id}`); } catch { continue; }   // one missed answer is not a failure
    if (up.status === "analysed") return { analysis_id: id };
    if (up.status === "failed") throw new Error(up.error || "The data could not be analysed.");
  }
  throw new Error("The analysis is taking too long. Try again in a minute.");
}
export const listUploads = () => call("/api/uploads");
export const excelUrl = (key) => `${API}/api/analysis/${key}/export.xlsx`;
export const getOutcomes = (key) => call(`/api/outcomes/${key}`);
export const ask = (key, question) => call(`/api/ask/${key}`, json({ question }));
