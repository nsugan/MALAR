import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import Card from "./Card.jsx";

export default function CoverageView({ coverage }) {
  const fields = coverage?.coverage || {};
  const rows = [];
  for (const [field, classes] of Object.entries(fields)) {
    for (const [cls, v] of Object.entries(classes)) {
      rows.push({ name: `${field}:${cls}`, value: v.value ?? 0, source: v.source });
    }
  }
  return (
    <Card title="Coverage / values (R per class)">
      {rows.length === 0 ? (
        <p className="text-xs text-slate-500">No values learned yet.</p>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={rows} margin={{ left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#94a3b8" }} interval={0} angle={-20} height={50} />
            <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
            <Bar dataKey="value" fill="#6366f1" />
          </BarChart>
        </ResponsiveContainer>
      )}
    </Card>
  );
}
