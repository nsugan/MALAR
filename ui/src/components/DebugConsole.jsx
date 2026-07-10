import { useEffect, useState, useCallback } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import InputInspector from "./InputInspector.jsx";

// A boxed panel for one section of internal state.
export function DebugBox({ title, children, tone = "slate" }) {
  const ring = { slate: "border-slate-200", indigo: "border-indigo-300",
    green: "border-emerald-300", amber: "border-amber-300" }[tone];
  return (
    <div className={`rounded-lg border ${ring} bg-slate-50 p-3`}>
      <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500">{title}</div>
      {children}
    </div>
  );
}

export function KV({ obj }) {
  return (
    <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[11px]">
      {Object.entries(obj || {}).map(([k, v]) => (
        <div key={k} className="contents">
          <span className="truncate text-slate-500">{k}</span>
          <span className="truncate text-slate-800">{fmt(v)}</span>
        </div>
      ))}
    </div>
  );
}
const fmt = (v) => (v && typeof v === "object" ? JSON.stringify(v) : String(v));

export function JsonBox({ data }) {
  return (
    <pre className="max-h-64 overflow-auto rounded bg-slate-50 p-2 text-[10px] leading-relaxed text-emerald-700">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

function Table({ rows, cols }) {
  if (!rows || rows.length === 0) return <p className="text-[11px] text-slate-400">empty</p>;
  return (
    <div className="max-h-56 overflow-auto">
      <table className="w-full text-left font-mono text-[11px]">
        <thead className="sticky top-0 bg-slate-100 text-slate-500">
          <tr>{cols.map((c) => <th key={c} className="pr-2">{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-slate-100">
              {cols.map((c) => <td key={c} className="truncate pr-2 text-slate-700">{fmt(r[c])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function DebugConsole() {
  const { activeId, debug } = useDomain();
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);

  const load = useCallback(() => {
    if (!activeId) return;
    api.debug(activeId).then(setD).catch((e) => setErr(String(e)));
  }, [activeId]);

  useEffect(() => {
    if (!debug || !activeId) return;
    load();
    const id = setInterval(load, 2000);     // live monitor
    return () => clearInterval(id);
  }, [debug, activeId, load]);

  if (!activeId) return <p className="text-sm text-slate-500">Select or create a domain first.</p>;
  if (!debug)
    return (
      <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-700">
        Debug mode is <b>off</b>. Tick the <b>Debug</b> checkbox in the header to stream the full
        internal state of every agent, gate, encoder and the curator (auto-refreshes every 2s).
      </div>
    );
  if (err) return <p className="text-sm text-rose-600">{err}</p>;
  if (!d) return <p className="text-sm text-slate-500">Loading internals…</p>;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-xs text-slate-500">
        <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" /> live · refreshes every 2s
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-700">Input inspector — examine a chosen file end-to-end</h3>
        <InputInspector />
      </div>

      <h3 className="mt-4 mb-1 text-sm font-semibold text-slate-700">Agents · gates · curator · memory (live)</h3>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <DebugBox title="Engine config / theorem guards" tone="indigo"><KV obj={flat(d.config)} /></DebugBox>
        <DebugBox title="Gates (conformal / OOD)" tone="amber"><KV obj={flat(d.gates?.conformal)} /></DebugBox>

        <DebugBox title="Curator agent (async)" tone="green">
          <p className="mb-1 text-[11px] text-slate-500">{d.curator?.description}</p>
          <KV obj={{ memory_size: d.curator?.memory_size, ...(d.curator?.config || {}) }} />
          <div className="mt-2 text-[10px] uppercase tracking-wider text-slate-500">decision audit (latest)</div>
          <Table rows={(d.curator?.audit_tail || []).slice().reverse()} cols={["op", "mem_id", "evicted"]} />
        </DebugBox>

        <DebugBox title="Last event (exact data received)" tone="indigo">
          <JsonBox data={d.last_event} />
        </DebugBox>

        <DebugBox title="Object agent — registry objects">
          <p className="mb-1 text-[11px] text-slate-500">{d.object_agent?.description}</p>
          <Table rows={d.object_agent?.objects}
            cols={["class", "provenance", "candidate", "phi_norm", "h_norm", "g_norm", "tau"]} />
        </DebugBox>

        <DebugBox title="Value agents A_k — {v_k,c}">
          <p className="mb-1 text-[11px] text-slate-500">{d.value_agents?.description}</p>
          <Table rows={d.value_agents?.records}
            cols={["objective_k", "class", "value", "source", "version", "sticky"]} />
        </DebugBox>

        <DebugBox title="Functional agents B_j — affordances">
          <p className="mb-1 text-[11px] text-slate-500">{d.functional_agents?.description}</p>
          <Table rows={d.functional_agents?.records}
            cols={["class", "dim", "action", "enabled", "source", "version"]} />
        </DebugBox>

        <DebugBox title="Encoders (topology / spectral / graph)">
          <JsonBox data={d.encoders} />
        </DebugBox>

        <DebugBox title="Memory items (vector norms)" tone="green">
          <Table rows={d.memory} cols={["id", "class", "omega", "tau", "phi_norm", "world_ctx"]} />
        </DebugBox>
      </div>
    </div>
  );
}

// flatten one level of nested objects for compact KV rendering
function flat(obj) {
  const out = {};
  for (const [k, v] of Object.entries(obj || {})) {
    if (v && typeof v === "object" && !Array.isArray(v))
      for (const [k2, v2] of Object.entries(v)) out[`${k}.${k2}`] = v2;
    else out[k] = v;
  }
  return out;
}
