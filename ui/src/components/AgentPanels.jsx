import { DebugBox, KV, JsonBox } from "./DebugConsole.jsx";

function Table({ rows, cols }) {
  if (!rows || rows.length === 0) return <p className="text-[11px] text-slate-400">none yet</p>;
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
const fmt = (v) => (v && typeof v === "object" ? JSON.stringify(v) : String(v));

// Renders a /debug snapshot grouped agent-by-agent.
export default function AgentPanels({ d, columns = 2 }) {
  if (!d) return <p className="text-sm text-slate-500">Loading agent state…</p>;
  return (
    <div className={`grid grid-cols-1 gap-3 ${columns === 2 ? "lg:grid-cols-2" : ""}`}>
      <DebugBox title="🧭 Perception / Encoders" >
        <JsonBox data={d.encoders} />
      </DebugBox>

      <DebugBox title="🔬 Object agent — identified objects" tone="indigo">
        <p className="mb-1 text-[11px] text-slate-500">{d.object_agent?.description}</p>
        <div className="mb-1 text-[11px] text-slate-600">classes: {(d.object_agent?.classes || []).join(", ") || "—"}</div>
        <Table rows={d.object_agent?.objects}
          cols={["class", "provenance", "candidate", "phi_norm", "h_norm", "g_norm", "tau"]} />
      </DebugBox>

      <DebugBox title="🎯 Value agents Aₖ — {v_k,c}" tone="amber">
        <p className="mb-1 text-[11px] text-slate-500">{d.value_agents?.description}</p>
        <Table rows={d.value_agents?.records}
          cols={["objective_k", "class", "value", "source", "version", "sticky"]} />
      </DebugBox>

      <DebugBox title="🛠 Functional agents Bⱼ — affordances" tone="green">
        <p className="mb-1 text-[11px] text-slate-500">{d.functional_agents?.description}</p>
        <Table rows={d.functional_agents?.records}
          cols={["class", "dim", "action", "enabled", "source", "version"]} />
      </DebugBox>

      <DebugBox title="🧮 Curator agent — decisions" tone="green">
        <p className="mb-1 text-[11px] text-slate-500">{d.curator?.description}</p>
        <KV obj={{ memory_size: d.curator?.memory_size, ...(d.curator?.config || {}) }} />
        <div className="mt-2 text-[10px] uppercase tracking-wider text-slate-500">audit (latest)</div>
        <Table rows={(d.curator?.audit_tail || []).slice().reverse()} cols={["op", "mem_id", "evicted"]} />
      </DebugBox>

      <DebugBox title="🚦 Critic / gates" tone="amber">
        <KV obj={d.gates?.conformal || {}} />
        <div className="mt-2 text-[10px] uppercase tracking-wider text-slate-500">memory items</div>
        <Table rows={d.memory} cols={["id", "class", "omega", "tau", "phi_norm"]} />
      </DebugBox>

      <DebugBox title="⏱ Last event (exact data received)" tone="indigo">
        <JsonBox data={d.last_event} />
      </DebugBox>

      <DebugBox title="⚙️ Engine config / theorem guards">
        <JsonBox data={d.config} />
      </DebugBox>
    </div>
  );
}
