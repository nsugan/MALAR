import { useState } from "react";
import Card from "./Card.jsx";
export default function HITLQueue({ pending, onLabel }) {
  const [label, setLabel] = useState("");
  return (
    <Card title="HITL label queue">
      {(!pending || pending.length === 0) ? (
        <p className="text-xs text-slate-500">No pending labels.</p>
      ) : (
        <div className="space-y-2">
          {pending.slice(0, 5).map((p) => (
            <div key={p.id} className="flex items-center gap-2">
              <span className="truncate text-[11px] text-slate-500">{p.id}</span>
              <input placeholder="class" value={label} onChange={(e) => setLabel(e.target.value)}
                className="w-24 rounded bg-white px-2 py-0.5 text-xs" />
              <button onClick={() => onLabel(p.id, label)}
                className="rounded bg-emerald-700 px-2 py-0.5 text-[11px]">Label</button>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
