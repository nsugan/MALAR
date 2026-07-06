import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";

export default function HyperspectralView({ spectrum = [], reconError }) {
  const data = spectrum.map((y, x) => ({ x, y }));
  return (
    <div>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
          <XAxis dataKey="x" tick={{ fontSize: 9, fill: "#475569" }} />
          <YAxis tick={{ fontSize: 9, fill: "#475569" }} />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 11 }} />
          <Line type="monotone" dataKey="y" stroke="#22d3ee" dot={false} strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>
      {reconError != null && (
        <div className="text-[10px] text-slate-500">recon error: {Number(reconError).toFixed(4)}</div>
      )}
    </div>
  );
}
