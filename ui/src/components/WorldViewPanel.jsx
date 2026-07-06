import Card from "./Card.jsx";
export default function WorldViewPanel({ memories }) {
  const classes = [...new Set((memories || []).map((m) => m.class))];
  return (
    <Card title="WorldView (grounded)">
      <p className="text-xs text-slate-500">
        {memories?.length || 0} active memories across {classes.length} regimes.
      </p>
      <div className="mt-2 flex flex-wrap gap-1">
        {classes.map((c) => (
          <span key={c} className="rounded bg-white px-2 py-0.5 text-[10px] text-slate-700">{c}</span>
        ))}
      </div>
    </Card>
  );
}
