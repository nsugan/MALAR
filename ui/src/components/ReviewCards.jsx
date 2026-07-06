import Card from "./Card.jsx";

const STAGES = ["perception", "topology", "identification", "value_learning",
  "functional_field", "fields_coordinator", "policy", "critic", "memory_write"];

export default function ReviewCards({ lastEvent, onDecision }) {
  const events = lastEvent?.events || [];
  return (
    <Card title="Per-stage review">
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {STAGES.map((s) => {
          const ev = events.find((e) => e.stage === s);
          return (
            <div key={s} className="rounded border border-slate-200 bg-slate-50 p-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-800">{s}</span>
                <span className={`h-2 w-2 rounded-full ${ev ? "bg-emerald-400" : "bg-slate-600"}`} />
              </div>
              <pre className="mt-1 max-h-20 overflow-auto text-[10px] text-slate-500">
                {ev ? JSON.stringify(stripStage(ev), null, 0) : "—"}
              </pre>
              {ev && (
                <div className="mt-1 flex gap-1">
                  <button onClick={() => onDecision(s, "approve")}
                    className="rounded bg-emerald-700 px-2 py-0.5 text-[10px]">Approve</button>
                  <button onClick={() => onDecision(s, "edit")}
                    className="rounded bg-amber-700 px-2 py-0.5 text-[10px]">Edit</button>
                  <button onClick={() => onDecision(s, "reject")}
                    className="rounded bg-rose-700 px-2 py-0.5 text-[10px]">Reject</button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}
function stripStage(e) { const { stage, t, ...rest } = e; return rest; }
