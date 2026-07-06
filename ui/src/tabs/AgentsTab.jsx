import { useEffect, useState, useCallback } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import Card, { Chip, Button } from "../components/Card.jsx";
import { KV } from "../components/DebugConsole.jsx";

const STORE_KEY = "malar_agent_toggles";

export default function AgentsTab() {
  const { activeId, debug } = useDomain();
  const [data, setData] = useState(null);
  const [auto, setAuto] = useState(true);
  const [on, setOn] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch { return {}; }
  });

  const load = useCallback(() => { if (activeId) api.agents(activeId).then(setData).catch(() => {}); }, [activeId]);
  useEffect(load, [load]);
  useEffect(() => {
    if (!auto || !activeId) return;
    const id = setInterval(load, 2000);
    return () => clearInterval(id);
  }, [auto, activeId, load]);

  const toggle = (k) => setOn((o) => {
    const next = { ...o, [k]: o[k] === false ? true : false };
    localStorage.setItem(STORE_KEY, JSON.stringify(next));
    return next;
  });
  const enabled = (k) => on[k] !== false;
  const setAll = (v) => {
    const next = {}; (data?.order || []).forEach((k) => { next[k] = v; });
    localStorage.setItem(STORE_KEY, JSON.stringify(next)); setOn(next);
  };

  if (!activeId) return <p className="text-sm text-slate-500">Select or create a domain first.</p>;
  if (!data) return <p className="text-sm text-slate-500">Loading agents…</p>;

  return (
    <div className="space-y-4">
      {!debug && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-2 text-xs text-amber-700">
          Tip: tick <b>Debug</b> in the header for the richest per-agent detail. The monitors below
          work either way.
        </div>
      )}
      <Card title="Agent monitors — toggle individual agents"
        right={
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-1 text-xs text-slate-500">
              <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> live
            </label>
            <Button variant="ghost" onClick={() => setAll(true)}>All on</Button>
            <Button variant="ghost" onClick={() => setAll(false)}>All off</Button>
          </div>
        }>
        <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3 lg:grid-cols-4">
          {data.order.map((k) => (
            <label key={k} className={`flex items-center gap-2 rounded border px-2 py-1 text-xs cursor-pointer
              ${enabled(k) ? "border-indigo-200 bg-indigo-50 text-indigo-700" : "border-slate-200 bg-white text-slate-500"}`}>
              <input type="checkbox" checked={enabled(k)} onChange={() => toggle(k)} />
              {data.agents[k].label}
            </label>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {data.order.filter(enabled).map((k) => (
          <AgentCard key={k} id={k} a={data.agents[k]} />
        ))}
      </div>
    </div>
  );
}

function AgentCard({ id, a }) {
  return (
    <Card title={a.label} right={<Chip color="indigo">{id}</Chip>}>
      <p className="mb-2 text-[11px] leading-relaxed text-slate-500">{a.role}</p>
      <KV obj={a.state} />
      {a.sub_agents && a.sub_agents.length > 0 && (
        <div className="mt-2 space-y-1.5">
          {a.sub_agents.map((sa, i) => (
            <div key={i} className="rounded border border-slate-200 bg-slate-50 p-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] font-semibold text-slate-700">{sa.name}</span>
                <Chip>{sa.n_classes} class{sa.n_classes === 1 ? "" : "es"}</Chip>
              </div>
              {sa.values && <ValueRow data={sa.values} />}
              {sa.affordances && (
                <div className="mt-1 space-y-0.5">
                  {Object.entries(sa.affordances).map(([cls, acts]) => (
                    <div key={cls} className="text-[11px]">
                      <span className="text-slate-600">{cls}: </span>
                      {acts.map((x) => <span key={x} className="mr-1 rounded bg-emerald-50 px-1 text-emerald-700">{x}</span>)}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      {a.activity && a.activity.length > 0 && (
        <div className="mt-2">
          <div className="text-[10px] uppercase tracking-wider text-slate-400">activity</div>
          <div className="max-h-32 overflow-auto font-mono text-[11px] text-slate-600">
            {a.activity.slice().reverse().map((e, i) => (
              <div key={i}>{e.op || JSON.stringify(e)}{e.mem_id ? ` · ${e.mem_id}` : ""}</div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

function ValueRow({ data }) {
  return (
    <div className="mt-1 grid grid-cols-2 gap-x-2 gap-y-0.5 font-mono text-[11px]">
      {Object.entries(data).map(([cls, v]) => (
        <div key={cls} className="flex justify-between">
          <span className="truncate text-slate-600">{cls}</span><span className="text-slate-800">{v}</span>
        </div>
      ))}
    </div>
  );
}
