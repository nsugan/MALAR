import { useEffect, useState } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import { Button, Chip } from "./Card.jsx";

// Live editor for all learning math / weights / parameters (per-domain).
export default function ParamEditor({ onApplied, compact }) {
  const { activeId } = useDomain();
  const [data, setData] = useState(null);
  const [edits, setEdits] = useState({});
  const [saved, setSaved] = useState(null);

  const load = () => { if (activeId) api.getParams(activeId).then((d) => { setData(d); setEdits({}); }); };
  useEffect(load, [activeId]);
  if (!data) return <p className="text-xs text-slate-500">Loading parameters…</p>;

  const set = (k, v) => setEdits((e) => ({ ...e, [k]: v }));
  const apply = async () => {
    const payload = {};
    for (const [k, v] of Object.entries(edits)) if (v !== "" && v != null) payload[k] = Number(v);
    const r = await api.setParams(activeId, payload);
    setSaved(r.applied); load();
    onApplied && onApplied(r.applied);
    setTimeout(() => setSaved(null), 2500);
  };
  const reset = () => setEdits({});

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Button variant="primary" onClick={apply}>Apply learning parameters</Button>
        <Button variant="ghost" onClick={reset}>Clear edits</Button>
        {saved && <Chip color="green">applied {Object.keys(saved).length} param(s)</Chip>}
        <span className="ml-auto text-[11px] text-slate-400">edits apply to the active domain's live engine</span>
      </div>
      <div className={`grid grid-cols-1 gap-3 ${compact ? "" : "lg:grid-cols-2"}`}>
        {Object.entries(data.groups).map(([group, fields]) => (
          <div key={group} className="rounded-lg border border-slate-200 bg-white p-3">
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">{group}</div>
            <div className="space-y-1.5">
              {Object.entries(fields).map(([k, v]) => (
                <div key={k} className="flex items-center gap-2" title={data.descriptions?.[k] || ""}>
                  <label className="w-40 truncate text-[12px] text-slate-600">{k}</label>
                  <input type="number" step="0.01"
                    value={edits[k] ?? v}
                    onChange={(e) => set(k, e.target.value)}
                    className="w-28 px-2 py-1 text-sm font-mono" />
                  {edits[k] != null && Number(edits[k]) !== Number(v) &&
                    <span className="text-[10px] text-amber-600">was {v}</span>}
                  <span className="truncate text-[10px] text-slate-400">{data.descriptions?.[k]}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
