import { num, pct } from "../format.js";

// Can I trust this? Data cleaning summary + model quality.
export default function DataHealth({ quality, model }) {
  const usedKey = Object.keys(quality).find((k) => k.endsWith("_used_for_analysis"));
  return (
    <section className="panel health">
      <div>
        <h2>Data health</h2>
        <p className="muted">{num(quality.rows_in_raw_file)} raw rows, {num(quality[usedKey])} used after cleaning.</p>
        <ul className="warnings">
          {quality.warnings.map((w) => (
            <li key={w.message} className={`warn-${w.level}`}><strong>{w.level === "warning" ? "Check" : "Note"}:</strong> {w.message}</li>
          ))}
        </ul>
      </div>
      <div>
        <h2>About the predictions</h2>
        <dl className="facts">
          <div><dt>Ranking quality (AUC)</dt><dd className="tnum">{model.test_auc}</dd></div>
          <div><dt>Problem rate, all items</dt><dd className="tnum">{pct(model.base_rate_test, 1)}</dd></div>
          <div><dt>Problem rate, top 10% flagged</dt><dd className="tnum">{pct(model.precision_top_10pct, 1)}</dd></div>
          <div><dt>Tested on</dt><dd className="tnum">{num(model.test_rows)} newest items</dd></div>
          {model.trained_on && <div><dt>Trained on</dt><dd>{model.trained_on}</dd></div>}
        </dl>
        <p className="muted small">{model.validation}.</p>
      </div>
    </section>
  );
}