// Persistence diagram (H1) — birth vs death scatter, with the diagonal.
export default function TopologyView({ diagram = [], summary = {}, size = 200 }) {
  const pts = (diagram || []).filter((d) => isFinite(d[0]) && isFinite(d[1]));
  const max = Math.max(1, ...pts.flat());
  const sc = (v) => (v / max) * (size - 24) + 8;
  return (
    <div>
      <svg width={size} height={size} className="rounded bg-slate-50">
        <line x1="8" y1={size - 8} x2={size - 8} y2="8" stroke="#cbd5e1" strokeDasharray="3 3" />
        {pts.map((d, i) => (
          <circle key={i} cx={sc(d[0])} cy={size - sc(d[1])} r="4" fill="#818cf8" opacity="0.85" />
        ))}
        <text x="8" y={size - 2} fontSize="8" fill="#64748b">birth →</text>
      </svg>
      <div className="mt-1 flex gap-2 text-[10px] text-slate-500">
        {["H0", "H1", "H2"].map((h) => summary[h] && (
          <span key={h}>{h}: {summary[h].count} (max {Number(summary[h].max_persistence).toFixed(2)})</span>
        ))}
      </div>
    </div>
  );
}
