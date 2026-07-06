import Card from "./Card.jsx";
export default function AuditLog({ log }) {
  return (
    <Card title="Agent activity / audit">
      <div className="max-h-40 space-y-1 overflow-auto text-[11px]">
        {(log || []).slice(-30).reverse().map((e, i) => (
          <div key={i} className="text-slate-500">
            <span className="text-slate-500">t{e.tick}</span> · {e.action || JSON.stringify(e).slice(0, 80)}
          </div>
        ))}
        {(!log || log.length === 0) && <p className="text-slate-400">No events yet.</p>}
      </div>
    </Card>
  );
}
