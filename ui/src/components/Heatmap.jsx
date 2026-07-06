// Compact matrix heatmap (SVG). Maps values to a blue→indigo→amber scale.
export default function Heatmap({ matrix, cellW = 6, cellH = 9, maxCols = 128, maxRows = 40, label }) {
  if (!matrix || matrix.length === 0) return <p className="text-[11px] text-slate-400">no data</p>;
  const rows = matrix.slice(0, maxRows);
  const cols = Math.min(maxCols, rows[0].length);
  let lo = Infinity, hi = -Infinity;
  for (const r of rows) for (let j = 0; j < cols; j++) { const v = r[j]; if (v < lo) lo = v; if (v > hi) hi = v; }
  const span = hi - lo || 1;
  const color = (v) => {
    const t = (v - lo) / span;                       // 0..1
    // light-friendly sequential: slate-50 -> indigo -> amber
    const r = Math.round(248 - 120 * t + 120 * Math.max(0, t - 0.6));
    const g = Math.round(250 - 180 * t);
    const b = Math.round(252 - 120 * t - 100 * Math.max(0, t - 0.6));
    return `rgb(${r},${g},${b})`;
  };
  return (
    <div>
      <svg width={cols * cellW} height={rows.length * cellH} className="rounded border border-slate-200">
        {rows.map((r, i) =>
          r.slice(0, cols).map((v, j) => (
            <rect key={`${i}-${j}`} x={j * cellW} y={i * cellH} width={cellW} height={cellH} fill={color(v)}>
              <title>{`row ${i}, col ${j} = ${v}`}</title>
            </rect>
          ))
        )}
      </svg>
      <div className="mt-1 flex items-center gap-2 text-[10px] text-slate-500">
        <span>{label || `${rows.length}×${cols}`}</span>
        <span className="ml-auto">min {lo.toFixed(3)}</span>
        <span className="inline-block h-2 w-16 rounded"
          style={{ background: "linear-gradient(90deg, rgb(248,250,252), rgb(128,70,180), rgb(248,160,60))" }} />
        <span>max {hi.toFixed(3)}</span>
      </div>
    </div>
  );
}
