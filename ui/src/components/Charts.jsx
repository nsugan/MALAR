import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from "recharts";

export function CoverageChart({ rows }) {
  if (!rows || rows.length === 0)
    return <p className="text-xs text-slate-500">No values learned yet.</p>;
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={rows} margin={{ left: -18, bottom: 30 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#475569" }} interval={0} angle={-25} textAnchor="end" height={50} />
        <YAxis tick={{ fontSize: 10, fill: "#475569" }} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
        <Bar dataKey="value">
          {rows.map((r, i) => <Cell key={i} fill={r.source === "human" ? "#f59e0b" : "#6366f1"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DistributionChart({ dist }) {
  const rows = Object.entries(dist || {}).map(([name, value]) => ({ name, value }));
  if (rows.length === 0) return <p className="text-xs text-slate-500">No results.</p>;
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={rows} margin={{ left: -18, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#475569" }} angle={-20} textAnchor="end" height={40} />
        <YAxis tick={{ fontSize: 10, fill: "#475569" }} allowDecimals={false} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
        <Bar dataKey="value" fill="#10b981" />
      </BarChart>
    </ResponsiveContainer>
  );
}
