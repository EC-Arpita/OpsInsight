import { ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceDot } from "recharts";
import { fmtDate } from "../format.js";

const BLUE = "#2a78d6";
const ORANGE = "#eb6834";
const GRID = "#e3e6e9";
const AXIS = { fontSize: 12, fill: "#56616d" };

function short(iso) {
  return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}

// Two separate charts (rate and volume): different units never share an axis.
export default function TrendCharts({ trends, words, anomalies }) {
  const rows = trends.series.map((p) => ({ ...p, rate: +(p[words.rateKey] * 100).toFixed(1), label: short(p.week_ending) }));
  const flagged = new Set(anomalies.map((a) => a.week_ending));
  return (
    <section className="charts">
      <figure className="chart">
        <figcaption>Weekly {words.problem} rate</figcaption>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={rows} margin={{ top: 10, right: 16, bottom: 0, left: -12 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="label" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} minTickGap={24} />
            <YAxis tick={AXIS} tickLine={false} axisLine={false} unit="%" />
            <Tooltip formatter={(v) => [`${v}%`, `${words.problem} rate`]} labelFormatter={(l, p) => `Week ending ${fmtDate(p?.[0]?.payload.week_ending)}`} />
            <Line type="monotone" dataKey="rate" stroke={ORANGE} strokeWidth={2} dot={false} activeDot={{ r: 5 }} />
            {rows.filter((r) => flagged.has(r.week_ending)).map((r) => (
              <ReferenceDot key={r.week_ending} x={r.label} y={r.rate} r={5} fill="#fff" stroke={ORANGE} strokeWidth={2} />
            ))}
          </LineChart>
        </ResponsiveContainer>
        {anomalies.length > 0 && (
          <p className="note">Circled weeks are unusual ({anomalies.length} found): up to {Math.max(...anomalies.map((a) => a.actual * 100)).toFixed(0)}% vs a normal {(anomalies[0].expected * 100).toFixed(0)}%.</p>
        )}
      </figure>
      <figure className="chart">
        <figcaption>Weekly {words.items}</figcaption>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={rows} margin={{ top: 10, right: 16, bottom: 0, left: -4 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="label" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} minTickGap={24} />
            <YAxis tick={AXIS} tickLine={false} axisLine={false} />
            <Tooltip formatter={(v) => [v.toLocaleString("en-IN"), words.items]} cursor={{ fill: "#eef1f3" }} />
            <Bar dataKey={words.countKey} fill={BLUE} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </figure>
    </section>
  );
}