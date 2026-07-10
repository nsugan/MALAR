import { useEffect, useState } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import Card, { Button, Chip } from "../components/Card.jsx";

export default function ConfigureTab() {
  const { activeId } = useDomain();
  const [cfg, setCfg] = useState(null);
  const [folder, setFolder] = useState("");
  const [saved, setSaved] = useState(false);
  const [plan, setPlan] = useState(null);
  const [planning, setPlanning] = useState(false);

  useEffect(() => { if (activeId) api.getConfig(activeId).then(setCfg); }, [activeId]);
  if (!activeId) return <Empty />;
  if (!cfg) return <p className="text-sm text-slate-500">Loading…</p>;

  const addFolder = () => { if (folder.trim()) { setCfg({ ...cfg, data_folders: [...(cfg.data_folders || []), folder.trim()] }); setFolder(""); } };
  const rmFolder = (f) => setCfg({ ...cfg, data_folders: cfg.data_folders.filter((x) => x !== f) });
  const runPlan = async (apply) => {
    setPlanning(true);
    try {
      const p = await api.planDomain(activeId, apply);
      setPlan(p);
      if (apply) { const fresh = await api.getConfig(activeId); setCfg(fresh); }
    } catch (e) {
      setPlan({ error: String(e), objectives: [], functionals: [], processing_steps: [], extra_agents: [] });
    }
    setPlanning(false);
  };
  const save = async () => {
    await api.putConfig(activeId, {
      data_folders: cfg.data_folders, data_description: cfg.data_description,
      objectives: cfg.objectives, functionals: cfg.functionals,
    });
    setSaved(true); setTimeout(() => setSaved(false), 1500);
  };
  const toggleAssist = async (v) => {
    setCfg({ ...cfg, llm_assist: v });
    await api.setLLMAssist(activeId, v);
  };

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Card title="Training data" right={saved && <Chip color="green">saved</Chip>}>
        <label className="text-xs text-slate-500">Data folders (server-visible paths)</label>
        <div className="mt-1 flex gap-2">
          <input value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="/app/data/<your-folder>"
            className="flex-1 px-2 py-1.5 text-sm" />
          <Button variant="ghost" onClick={addFolder}>Add</Button>
        </div>
        <div className="mt-2 space-y-1">
          {(cfg.data_folders || []).map((f) => (
            <div key={f} className="flex items-center justify-between rounded bg-slate-50 px-2 py-1 text-xs">
              <span className="truncate text-slate-700">{f}</span>
              <button onClick={() => rmFolder(f)} className="text-rose-600">✕</button>
            </div>
          ))}
          {(!cfg.data_folders || cfg.data_folders.length === 0) &&
            <p className="text-[11px] text-slate-500">None — training will use built-in synthetic data.</p>}
        </div>
        <label className="mt-3 block text-xs text-slate-500">Data description — what the data is, its
          classes/labels, format and source (this is where the domain details live; used as LLM context)</label>
        <textarea value={cfg.data_description || ""} onChange={(e) => setCfg({ ...cfg, data_description: e.target.value })}
          placeholder="e.g. CSV feature vectors per sample, folder-per-class; 3 classes; from …"
          className="mt-1 h-28 w-full px-2 py-1.5 text-sm" />
        <div className="mt-2 flex items-center gap-2">
          <Button onClick={save}>Save config</Button>
          <label className="flex items-center gap-2 text-xs text-slate-600">
            <input type="checkbox" checked={!!cfg.llm_assist} onChange={(e) => toggleAssist(e.target.checked)} />
            LLM-assist per item (slower)
          </label>
        </div>
      </Card>

      <Card title="DomainSpec">
        <Section title="Objectives R">
          {(cfg.objectives || []).map((o, i) => (
            <div key={i} className="flex items-center justify-between text-sm">
              <span className="text-slate-800">{o.key}</span>
              <Chip>target {o.target ?? "—"}</Chip>
            </div>
          ))}
        </Section>
        <Section title="Functional dims F">
          {(cfg.functionals || []).map((f, i) => (
            <div key={i} className="text-sm">
              <span className="text-slate-800">{f.dim}: </span>
              <span className="text-slate-500">{(f.actions || []).join(", ")}</span>
            </div>
          ))}
        </Section>
        <Section title="Coverage targets">
          <pre className="text-[11px] text-slate-500">{JSON.stringify(cfg.coverage_targets, null, 1)}</pre>
        </Section>
      </Card>

      <Card title="Orchestrator — LLM plan from the data description" className="lg:col-span-2">
        <p className="text-xs text-slate-500">
          The Orchestrator/Planner sends your <b>data description</b> to the LLM to derive the objective
          <b> R</b> fields, functional <b>F</b> dims (actions), processing steps, and any extra
          domain-specific agents — then you can apply it to the DomainSpec.
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Button disabled={planning} onClick={() => runPlan(false)}>{planning ? "Planning… (first call can take ~2 min)" : "Plan with LLM (preview)"}</Button>
          <Button variant="green" disabled={planning} onClick={() => runPlan(true)}>Plan &amp; apply to DomainSpec</Button>
          {plan && !plan.error && <Chip color={plan.used_llm ? "green" : "amber"}>{plan.used_llm ? "from LLM" : "fallback (gateway off / timed out)"}</Chip>}
          {plan && plan.error && <Chip color="red">error</Chip>}
        </div>
        <p className="mt-1 text-[11px] text-slate-400">
          The first LLM call can take up to ~2 min while the local model loads into memory; later calls are fast.
        </p>
        {plan && plan.error && <p className="mt-2 text-xs text-rose-600">{plan.error}</p>}
        {plan && !plan.error && (
          <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
            <PlanBox title="Objectives R">
              {plan.objectives.map((o, i) => (
                <div key={i} className="text-sm"><b className="text-slate-700">{o.key}</b>
                  <span className="text-slate-500"> · {o.direction} · target {o.target}</span>
                  <div className="text-[11px] text-slate-500">{o.description}</div></div>
              ))}
            </PlanBox>
            <PlanBox title="Functional dims F (actions)">
              {plan.functionals.map((f, i) => (
                <div key={i} className="text-sm"><b className="text-slate-700">{f.dim}</b>:
                  {" "}{(f.actions || []).map((a) => <span key={a} className="mr-1 rounded bg-emerald-50 px-1 text-[11px] text-emerald-700">{a}</span>)}</div>
              ))}
            </PlanBox>
            <PlanBox title="Processing steps">
              <ol className="list-decimal pl-4 text-[12px] text-slate-600">{(plan.processing_steps || []).map((s2, i) => <li key={i}>{s2}</li>)}</ol>
            </PlanBox>
            <PlanBox title="Extra agents (LLM-proposed)">
              {(plan.extra_agents || []).length === 0 ? <span className="text-[11px] text-slate-400">none</span> :
                plan.extra_agents.map((e, i) => (
                  <div key={i} className="text-sm"><b className="text-slate-700">{e.name}</b>
                    <div className="text-[11px] text-slate-500">{e.role}</div></div>))}
            </PlanBox>
            {plan.rationale && <div className="lg:col-span-2 text-[12px] italic text-slate-500">{plan.rationale}</div>}
          </div>
        )}
      </Card>
    </div>
  );
}
const Section = ({ title, children }) => (
  <div className="mb-3"><div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">{title}</div>{children}</div>
);
const PlanBox = ({ title, children }) => (
  <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
    <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">{title}</div>
    <div className="space-y-1">{children}</div>
  </div>
);
const Empty = () => <p className="text-sm text-slate-500">Select or create a domain first.</p>;
