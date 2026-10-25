import { useState } from "react";
import { ResponsiveContainer, ComposedChart, Bar, Cell, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { fmtDate, num } from "../format.js";

const BLUE = "#2a78d6";
const ORANGE = "#eb6834";
const GRID = "#e3e6e9";
const AXIS = { fontSize: 12, fill: "#56616d" };

const day = (iso) => new Date(iso).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });

// Works for both domains: tickets come per team with a capacity, orders per region with a usual level.
export default function Forecast({ forecast }) {
  const isTickets = Array.isArray(forecast?.teams);
  const groups = isTickets ? forecast.teams : forecast?.regions;
  const [pick, setPick] = useState(0);
  if (!groups || groups.length === 0) {
    return <p className="muted pad">No forecast in this analysis yet. Run python -m opsinsight.forecast.run in the api folder.</p>;
  }
  const g = groups[Math.min(pick, groups.length - 1)];
  const name = (x) => x.team || x.region;
  const words = isTickets
    ? { unit: "tickets", limit: "Capacity", flag: "overload", flagged: "Overloaded days", total: "expected_tickets_7d", limit7: "expected_capacity_7d", flags: "overloaded_days" }
    : { unit: "orders", limit: "Usual", flag: "busy", flagged: "Busy days", total: "expected_orders_7d", limit7: "usual_orders_7d", flags: "busy_days" };
  const rows = g.daily.map((d) => ({
    label: day(d.date), date: d.date,
    expected: isTickets ? d.expected_tickets : d.expected_orders,
    limit: isTickets ? d.expected_capacity : d.usual_orders,
    flag: d[words.flag],
  }));
  const flagged = rows.filter((r) => r.flag);

  return (
    <section>
      <div className="two-col">
        <figure className="chart">
          <div className="panel-head">
            <figcaption>Next 7 days: expected {words.unit}, {name(g)}</figcaption>
            <div className="mini-tabs" role="group" aria-label="Choose a group">
              {groups.map((x, i) => (
                <button key={name(x)} className={i === pick ? "chip on" : "chip"} aria-pressed={i === pick} onClick={() => setPick(i)}>
                  {name(x)}
                </button>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={rows} margin={{ top: 10, right: 16, bottom: 0, left: -4 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="label" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
              <YAxis tick={AXIS} tickLine={false} axisLine={false} />
              <Tooltip formatter={(v, k) => [num(v), k === "expected" ? `expected ${words.unit}` : words.limit.toLowerCase()]} labelFormatter={(l, p) => fmtDate(p?.[0]?.payload.date)} cursor={{ fill: "#eef1f3" }} />
              <Bar dataKey="expected" radius={[4, 4, 0, 0]}>
                {rows.map((r) => <Cell key={r.date} fill={r.flag ? ORANGE : BLUE} />)}
              </Bar>
              <Line dataKey="limit" type="linear" stroke="#18212b" strokeDasharray="4 4" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <p className="note">
            Dashed line: {isTickets ? "capacity: what the team usually resolves on that weekday" : "the usual level for that weekday"}.{" "}
            {flagged.length === 0
              ? `No ${words.flag} days expected for ${name(g)}.`
              : `${words.flagged} (orange): ${flagged.map((r) => r.label).join(", ")}.`}{" "}
            Rule: {isTickets ? forecast.overload_rule : forecast.busy_rule}.
          </p>
        </figure>

        <div className="panel">
          <div className="panel-head"><strong>All {isTickets ? "teams" : "regions"}, next 7 days</strong></div>
          <table className="table">
            <thead>
              <tr><th>{isTickets ? "Team" : "Region"}</th><th className="r">Expected</th><th className="r">{isTickets ? "Capacity" : "Usual week"}</th><th className="r">{words.flagged}</th></tr>
            </thead>
            <tbody>
              {groups.map((x) => (
                <tr key={name(x)}>
                  <td>{name(x)}</td>
                  <td className="r tnum">{num(x[words.total])}</td>
                  <td className="r tnum">{num(x[words.limit7])}</td>
                  <td className={x[words.flags] > 0 ? "r tnum flag" : "r tnum"}>{x[words.flags]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">
            Method: <strong>{forecast.method}</strong>, chosen because it had the smallest error when tested on the last two weeks
            {forecast.backtest_mean_abs_error ? ` (${forecast.backtest_mean_abs_error[0].mean_abs_error} ${words.unit} per day, against ${forecast.backtest_mean_abs_error.at(-1).mean_abs_error} for the worst method)` : ""}.
          </p>
          {forecast.note && <p className="note">{forecast.note}</p>}
        </div>
      </div>
    </section>
  );
}
