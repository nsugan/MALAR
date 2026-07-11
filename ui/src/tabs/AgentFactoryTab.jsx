import { useEffect, useState } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import Card, { Button, Chip } from "../components/Card.jsx";

const SOURCES = [
  ["planner", "Planner (LLM-proposed)"],
  ["functional", "Functional-Field agents (Bⱼ)"],
  ["objective", "Objective-Field agents (Aⱼ)"],
  ["value", "Value-Field agents"],
];

function TraceBlock({ trace }) {
  let t = null;
  try { t = trace ? JSON.parse(trace) : null; } catch (e) { t = null; }
  if (!t) return null;
  const Sec = ({ title, ex }) => (
    <details className="rounded border border-slate-200 bg-slate-50">
      <summary className="cursor-pointer px-2 py-1 text-[11px] text-slate-600">
        {title} <span className="text-slate-400">· via {ex?.alias || "?"}</span>
      </summary>
      <div className="space-y-1 px-2 pb-2">
        <div className="text-[9px] uppercase tracking-wider text-slate-400">query sent</div>
        <pre className="max-h-32 overflow-auto whitespace-pre-wrap rounded bg-white p-1.5 text-[10px] text-slate-600">{ex?.prompt || "—"}</pre>
        <div className="text-[9px] uppercase tracking-wider text-slate-400">response received</div>
        <pre className={`max-h-40 overflow-auto whitespace-pre-wrap rounded p-1.5 text-[10px] ${String(ex?.response || "").startsWith("ERROR") ? "bg-rose-50 text-rose-700" : "bg-white text-slate-700"}`}>{ex?.response || "(no response)"}</pre>
      </div>
    </details>
  );
  return (
    <div className="space-y-1">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">LLM exchange (what each agent was asked & answered)</div>
      <Sec title="1 · Algorithm agent" ex={t.algo} />
      <Sec title="2 · Code agent" ex={t.code} />
    </div>
  );
}

function IOBlock({ input, output, ts }) {
  const parse = (s) => {
    if (s == null) return null;
    try { return typeof s === "string" ? JSON.parse(s) : s; } catch (e) { return s; }
  };
  const inp = parse(input);
  const outp = parse(output);
  if (inp == null && outp == null) {
    return (
      <div className="rounded border border-slate-200 bg-slate-50 px-2 py-1.5 text-[11px] text-slate-500">
        No runs recorded yet — click <b>Test on domain sample</b> (or <b>Run now</b>) to see the
        exact input this agent received and the output it produced.
      </div>
    );
  }
  const when = ts ? new Date(ts * 1000).toLocaleString() : "";
  const failed = outp && typeof outp === "object" && outp.ok === false;
  return (
    <div className="rounded-lg border border-indigo-200 bg-indigo-50/40 p-2">
      <div className="mb-1 flex items-center justify-between">
        <div className="text-[10px] uppercase tracking-wider text-indigo-600">
          Last run — input &amp; output (verify the agent does its task)
        </div>
        {when && <span className="text-[10px] text-slate-400">{when}</span>}
      </div>
      <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
        <div>
          <div className="text-[9px] uppercase tracking-wider text-slate-400">input (ctx it received)</div>
          <pre className="max-h-44 overflow-auto whitespace-pre-wrap rounded bg-white p-1.5 text-[10px] text-slate-700">{JSON.stringify(inp, null, 1)}</pre>
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-wider text-slate-400">output (what it produced)</div>
          <pre className={`max-h-44 overflow-auto whitespace-pre-wrap rounded p-1.5 text-[10px] ${failed ? "bg-rose-50 text-rose-700" : "bg-white text-slate-700"}`}>{JSON.stringify(outp, null, 1)}</pre>
        </div>
      </div>
    </div>
  );
}

export default function AgentFactoryTab() {
  const { activeId } = useDomain();
  const [source, setSource] = useState("planner");
  const [proposals, setProposals] = useState([]);
  const [chosen, setChosen] = useState({});
  const [agents, setAgents] = useState([]);
  const [sel, setSel] = useState(null);
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [out, setOut] = useState(null);
  const [onlyVal, setOnlyVal] = useState(true);
  const [crossDomain, setCrossDomain] = useState(false);
  const [activation, setActivation] = useState(null);
  const [useInLoop, setUseInLoop] = useState(false);

  const refresh = async () => {
    try { setAgents((await api.listAgents(activeId, crossDomain)).agents || []); }
    catch (e) { setMsg(String(e)); }
  };
  useEffect(() => { refresh(); }, [activeId, crossDomain]);
  useEffect(() => { if (activeId) api.agentsStatus(activeId).then((r) => setUseInLoop(!!r.agents_active)).catch(() => {}); }, [activeId]);

  const toggleInLoop = async (v) => {
    setUseInLoop(v);
    try { await api.useAgents(activeId, v); setMsg(v ? 'validated agents will run during training (fed back to parent fields, stored to DB)' : 'in-loop agents disabled'); }
    catch (e) { setMsg('toggle failed: ' + e); }
  };

  const activate = async () => {
    if (!activeId) { setMsg('select a domain first'); return; }
    setBusy(true); setMsg('running validated agents…');
    try { const r = await api.activateAgents(activeId, onlyVal, crossDomain); setActivation(r);
      setMsg(`activated ${r.activated} agent(s) → stored in ${r.stored_in}`); }
    catch (e) { setMsg('activate failed: ' + e); }
    setBusy(false);
  };

  const loadProposals = async () => {
    if (!activeId) { setMsg("select a domain first"); return; }
    setMsg(""); setProposals([]); setChosen({});
    try {
      const r = await api.agentProposals(activeId, source);
      setProposals(r.proposals || []);
      if (!(r.proposals || []).length) setMsg("no proposals from this source — run Plan / train first");
    } catch (e) { setMsg("failed: " + e); }
  };
  const toggle = (n) => setChosen((c) => ({ ...c, [n]: !c[n] }));
  const allOn = () => setChosen(Object.fromEntries(proposals.map((p) => [p.name, true])));
  const generate = async () => {
    const picks = proposals.filter((p) => chosen[p.name]);
    if (!picks.length) { setMsg("select at least one proposal"); return; }
    setBusy(true);
    for (let i = 0; i < picks.length; i++) {
      setMsg(`generating ${i + 1}/${picks.length}: ${picks[i].name}…  (algorithm + code, may take a minute)`);
      try {
        // generate one at a time via the domain endpoint so a real data sample is fed to the coder + auto-tested
        await fetch(`/api/domains/${activeId}/agents/generate-selected`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ agents: [{ name: picks[i].name, role: picks[i].role }] }),
        });
      } catch (e) { setMsg(`error on ${picks[i].name}: ${e}`); }
      await refresh();
    }
    setMsg(`done: ${picks.length} agent(s) generated & auto-tested`);
    setBusy(false);
  };

  const open = async (aid) => {
    setOut(null);
    try { const a = await api.getAgent(aid); setSel(a); setCode(a.code || ""); } catch (e) { setMsg(String(e)); }
  };
  const saveCode = async () => {
    setBusy(true); setMsg("");
    try {
      const r = await api.updateAgentCode(sel.id, code, activeId);
      setOut(r);
      setMsg(r.validation?.ok ? (r.test?.ok ? "saved · valid · test passed" : "saved · valid · TEST ERROR")
                              : "saved · validation issues");
      await refresh(); const a = await api.getAgent(sel.id); setSel(a);
    } catch (e) { setMsg("save failed: " + e); }
    setBusy(false);
  };
  const testReal = async () => {
    setOut("testing on domain sample…");
    try {
      setOut(await api.testAgent(sel.id, { did: activeId }));
      const a = await api.getAgent(sel.id); setSel(a);   // refresh last input/output
    } catch (e) { setOut({ ok: false, error: String(e) }); }
  };
  const validate = async (v) => { await api.validateAgent(sel.id, v); await refresh(); const a = await api.getAgent(sel.id); setSel(a); };
  const removeAgent = async () => {
    if (!confirm(`Delete agent '${sel.name}'? This removes its code from agent memory permanently.`)) return;
    try { await api.deleteAgent(sel.id); setSel(null); setCode(""); setOut(null); setMsg(`deleted ${sel.name}`); await refresh(); }
    catch (e) { setMsg('delete failed: ' + e); }
  };

  const nChosen = proposals.filter((p) => chosen[p.name]).length;
  // an agent owned by ANOTHER domain (shown via cross-domain) is read-only here
  const foreign = !!(sel && sel.domain_id && activeId && sel.domain_id !== activeId);

  return (
    <div className="space-y-4">
      <Card title="Choose which agents to create">
        <p className="text-xs text-slate-500">
          Pick a source, review each proposal, tick the ones to build, then generate. The coder
          LLM is given a <b>real data sample</b> from this domain and each agent is <b>auto-tested</b>
          on it — so syntax/runtime errors show up immediately. You can then edit, test and save.
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <select value={source} onChange={(e) => setSource(e.target.value)} className="rounded bg-white px-2 py-1.5 text-sm">
            {SOURCES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
          <Button variant="ghost" onClick={loadProposals}>Load proposals</Button>
          {proposals.length > 0 && <>
            <Button variant="ghost" onClick={allOn}>Select all</Button>
            <Button variant="green" disabled={busy || nChosen === 0} onClick={generate}>
              {busy ? "Working…" : `Generate selected (${nChosen})`}</Button>
          </>}
          {msg && <span className="text-[11px] text-slate-600">{msg}</span>}
        </div>
        {proposals.length > 0 && (
          <div className="mt-3 max-h-64 space-y-1 overflow-auto">
            {proposals.map((p) => (
              <label key={p.name} className="flex cursor-pointer items-start gap-2 rounded border border-slate-200 bg-slate-50 px-2 py-1.5">
                <input type="checkbox" className="mt-1" checked={!!chosen[p.name]} onChange={() => toggle(p.name)} />
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium text-slate-800">{p.name} <Chip>{p.source}</Chip></div>
                  <div className="text-[11px] text-slate-500">{p.role}</div>
                  {p.proposed && <div className="text-[10px] text-slate-400">proposed: {JSON.stringify(p.proposed)}</div>}
                </div>
              </label>
            ))}
          </div>
        )}
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
        <Card title={`Agents in this domain (${agents.filter((a) => !a.cross_domain).length})`} className="lg:col-span-2"
          right={
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-1 text-[11px] text-slate-600" title="Show other domains' VALIDATED agents too, so the orchestrator can use them together for cross-domain inference.">
                <input type="checkbox" checked={crossDomain} onChange={(e) => setCrossDomain(e.target.checked)} /> cross-domain
              </label>
              <label className="flex items-center gap-1 text-[11px] text-slate-600" title="Parent field-agents run their validated agents on every training item; outputs feed back into the fields and are stored in the DB.">
                <input type="checkbox" checked={useInLoop} onChange={(e) => toggleInLoop(e.target.checked)} /> use in training
              </label>
              <label className="flex items-center gap-1 text-[11px] text-slate-600">
                <input type="checkbox" checked={onlyVal} onChange={(e) => setOnlyVal(e.target.checked)} /> only validated
              </label>
              <Button variant="green" disabled={busy} onClick={activate}>Run now</Button>
            </div>
          }>
          <div className="space-y-1">
            {agents.length === 0 && <p className="text-[11px] text-slate-400">None yet.</p>}
            {agents.map((a) => (
              <div key={a.id} onClick={() => open(a.id)}
                className={`cursor-pointer rounded border px-2 py-1.5 text-sm
                  ${sel && sel.id === a.id ? "border-indigo-400 bg-indigo-50" : "border-slate-200 bg-slate-50"}`}>
                <div className="flex items-center justify-between">
                  <span className="font-medium text-slate-800">{a.name}</span>
                  <div className="flex items-center gap-1.5">
                    {a.cross_domain && <Chip color="amber" title={`from domain ${a.domain_id}`}>⤳ {a.domain_id}</Chip>}
                    <Chip color={a.validated ? "green" : "amber"}>{a.validated ? "validated" : "pending"}</Chip>
                    <Chip>used {a.usage_count}</Chip>
                  </div>
                </div>
                {a.note && <div className={`text-[10px] ${String(a.note).includes("ERROR") || String(a.note).startsWith("invalid") ? "text-rose-600" : "text-slate-400"}`}>{a.note}</div>}
              </div>
            ))}
          </div>
          {activation && (
            <div className="mt-2 border-t border-slate-200 pt-2">
              <div className="text-[10px] uppercase tracking-wider text-slate-500">Activation — outputs routed to parent agents · stored in {activation.stored_in}</div>
              <div className="mt-1 max-h-40 space-y-1 overflow-auto">
                {(activation.results || []).map((r, i) => (
                  <div key={i} className="rounded bg-slate-50 px-2 py-1 text-[11px]">
                    <b className="text-slate-700">{r.agent}</b> <span className="text-slate-400">→ {r.parent}</span>
                    {r.ok ? <span className="ml-1 text-emerald-600">{JSON.stringify(r.output)}</span>
                          : <span className="ml-1 text-rose-600">{r.error}</span>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>

        <Card title={sel ? `Edit agent: ${sel.name}` : "Select an agent"} className="lg:col-span-3">
          {!sel ? <p className="text-sm text-slate-500">Pick a stored agent to edit and test its code.</p> : (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <Chip color={sel.validated ? "green" : "amber"}>{sel.validated ? "validated" : "pending review"}</Chip>
                {foreign && <Chip color="amber" title={`owned by domain ${sel.domain_id}`}>⤳ from {sel.domain_id} · read-only here</Chip>}
                {sel.note && <Chip color={String(sel.note).includes("ERROR") || String(sel.note).startsWith("invalid") ? "red" : "green"}>{sel.note}</Chip>}
                <Button disabled={busy || foreign} onClick={saveCode}>Save code</Button>
                <Button variant="ghost" onClick={testReal}>Test on domain sample</Button>
                {!sel.validated
                  ? <Button variant="green" disabled={foreign} onClick={() => validate(true)}>✓ Validate</Button>
                  : <Button variant="ghost" disabled={foreign} onClick={() => validate(false)}>Revoke</Button>}
                <Button variant="red" disabled={foreign} onClick={removeAgent}>Delete</Button>
              </div>
              <IOBlock input={sel.last_input} output={sel.last_output} ts={sel.last_run} />
              {sel.algorithm && <details className="text-[11px] text-slate-600">
                <summary className="cursor-pointer text-slate-500">algorithm (design)</summary>
                <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap rounded bg-slate-50 p-2">{sel.algorithm}</pre>
              </details>}
              <TraceBlock trace={sel.trace} />
              <div>
                <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">Code (editable — saving re-validates & re-tests, resets to pending)</div>
                <textarea value={code} onChange={(e) => setCode(e.target.value)} spellCheck={false}
                  className="h-72 w-full rounded bg-slate-900 p-2 font-mono text-[11px] text-emerald-200" />
              </div>
              {out && (
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">Result</div>
                  <pre className="max-h-40 overflow-auto rounded bg-slate-50 p-2 text-[11px] text-slate-700">{typeof out === "string" ? out : JSON.stringify(out, null, 1)}</pre>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
