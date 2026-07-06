import { useEffect, useState } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import Card, { Chip } from "../components/Card.jsx";
import { CoverageChart } from "../components/Charts.jsx";

export default function KnowledgeTab() {
  const { activeId } = useDomain();
  const [values, setValues] = useState({});
  const [affs, setAffs] = useState({});

  const load = () => {
    if (!activeId) return;
    api.learnedValues(activeId).then((r) => setValues(r.values || {}));
    api.learnedAffordances(activeId).then((r) => setAffs(r.affordances || {}));
  };
  useEffect(load, [activeId]);
  if (!activeId) return <p className="text-sm text-slate-500">Select or create a domain first.</p>;

  const rows = [];
  for (const [field, classes] of Object.entries(values))
    for (const [cls, v] of Object.entries(classes))
      rows.push({ name: `${field}:${cls}`, value: v.value, source: v.source });

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Card title="Objective values  {v_k,c}  (orange = human-set, sticky)">
        <CoverageChart rows={rows} />
      </Card>
      <Card title="Per-class detail">
        {Object.keys(values).length === 0 ? <p className="text-xs text-slate-500">Nothing learned yet — train first.</p> : (
          <div className="space-y-2">
            {Object.entries(values).map(([field, classes]) => (
              <div key={field}>
                <div className="text-[10px] uppercase tracking-wider text-slate-500">{field}</div>
                {Object.entries(classes).map(([cls, v]) => (
                  <div key={cls} className="flex items-center justify-between text-sm">
                    <span className="text-slate-800">{cls}</span>
                    <span className="flex items-center gap-2">
                      <Chip color={v.source === "human" ? "amber" : "indigo"}>{v.source} v{v.version}</Chip>
                      <span className="font-mono text-slate-700">{v.value}</span>
                    </span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </Card>
      <Card title="Affordances F" className="lg:col-span-2">
        {Object.keys(affs).length === 0 ? <p className="text-xs text-slate-500">No affordances learned yet.</p> : (
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            {Object.entries(affs).map(([cls, list]) => (
              <div key={cls} className="rounded border border-slate-200 bg-slate-50 p-2">
                <div className="text-sm font-semibold text-slate-800">{cls}</div>
                <div className="mt-1 flex flex-wrap gap-1">
                  {list.map((a, i) => <Chip key={i} color="green">{a.action}</Chip>)}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
