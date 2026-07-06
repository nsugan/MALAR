import { useEffect, useState } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import Card, { Button, Chip } from "../components/Card.jsx";
import GraphView from "../components/GraphView.jsx";
import TopologyView from "../components/TopologyView.jsx";
import HyperspectralView from "../components/HyperspectralView.jsx";
import { DebugBox, JsonBox } from "../components/DebugConsole.jsx";
import AgentPanels from "../components/AgentPanels.jsx";
import ParamEditor from "../components/ParamEditor.jsx";

export default function TrainTab() {
  const { activeId, mode, setMode } = useDomain();
  if (!activeId) return <p className="text-sm text-slate-500">Select or create a domain first.</p>;
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500">Mode:</span>
        <Button variant={mode === "supervised" ? "primary" : "ghost"} onClick={() => setMode("supervised")}>Supervised (one-at-a-time)</Button>
        <Button variant={mode === "unsupervised" ? "primary" : "ghost"} onClick={() => setMode("unsupervised")}>Unsupervised (auto)</Button>
      </div>
      {mode === "supervised" ? <Supervised did={activeId} /> : <Unsupervised did={activeId} />}
    </div>
  );
}

function Supervised({ did }) {
  const { refresh, debug } = useDomain();
  const [pv, setPv] = useState(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [src, setSrc] = useState(null);
  const [dbg, setDbg] = useState(null);
  const [assist, setAssist] = useState(false);
  const [correcting, setCorrecting] = useState(false);
  const refreshDebug = () => { if (debug) api.debug(did).then(setDbg); };

  const load = async () => {
    setBusy(true);
    const nxt = await api.trainNext(did);
    if (nxt.done) { setDone(true); setPv(null); setBusy(false); return; }
    setPv(await api.trainPreview(did));
    refreshDebug();
    setBusy(false);
  };
  useEffect(() => { api.trainStart(did).then(load); api.inputs(did).then(setSrc); api.getConfig(did).then((c) => setAssist(!!c.llm_assist)); /* eslint-disable-next-line */ }, [did]);

  const act = async (decision, corrections) => {
    setBusy(true);
    await api.trainConfirm(did, { decision, corrections });
    refresh(); refreshDebug();
    await load();
  };

  if (done) return <Card title="Supervised training"><p className="text-sm text-emerald-700">All items reviewed. Switch to Knowledge or World Graph to inspect what was learned.</p>
    <Button className="mt-2" onClick={() => { setDone(false); api.trainStart(did).then(load); }}>Restart queue</Button></Card>;
  if (!pv) return <p className="text-sm text-slate-500">Loading item…</p>;

  return (
    <div className="space-y-3">
      {src && (
        <div className={`rounded-lg border p-2 text-xs ${src.source === "folder"
          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
          : src.source === "folder_empty" ? "border-rose-200 bg-rose-50 text-rose-700"
          : "border-amber-200 bg-amber-50 text-amber-700"}`}>
          <b>{src.total} items</b> · source: <b>{src.source}</b> — {src.note}
        </div>
      )}
      <label className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white p-2 text-xs text-slate-600">
        <input type="checkbox" checked={assist}
          onChange={async (e) => { setAssist(e.target.checked); await api.setLLMAssist(did, e.target.checked); }} />
        <b>LLM-assist identification</b> — call the LLM per item to propose the object class &amp;
        functionalities (slower; the first call can take up to ~2&nbsp;min while the local model loads).
        Off = fast deterministic identification.
      </label>
      <div className="flex items-center gap-2 text-sm text-slate-700">
        <Chip color="indigo">item {pv.index + 1}/{pv.total}</Chip>
        <span className="font-mono text-slate-500">{pv.item_id}</span>
        <span className="ml-auto">Proposed: <b className="text-indigo-700">{pv.identification.proposed}</b>
          <span className="ml-1 text-slate-500">({pv.identification.method}, score {pv.identification.score})</span></span>
      </div>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        <Card title="World graph (region)"><GraphView data={pv.graph} height={200} /></Card>
        <Card title="Topology (persistence)"><TopologyView diagram={pv.topology.h1_diagram} summary={pv.topology.summary} /></Card>
        <Card title="Hyperspectral"><HyperspectralView spectrum={pv.hyper.spectrum} reconError={pv.hyper.recon_error} /></Card>
      </div>
      <Card title="Proposed knowledge">
        <div className="flex flex-wrap gap-4 text-sm">
          <div><span className="text-slate-500">Values R:</span> {Object.entries(pv.values).map(([k, v]) => <Chip key={k}>{k}={v}</Chip>)}</div>
          <div><span className="text-slate-500">Affordances F:</span> {pv.affordances.map((a) => <Chip key={a} color="green">{a}</Chip>)}</div>
        </div>
        <div className="mt-3 flex items-center gap-2">
          <Button variant="green" disabled={busy} onClick={() => act("confirm")}>Confirm — learned correctly</Button>
          <Button variant="amber" disabled={busy} onClick={() => setCorrecting((v) => !v)}>
            {correcting ? "Close correction" : "Correct — edit label / values / learning math"}
          </Button>
          <Button variant="ghost" disabled={busy} onClick={() => act("skip")}>Skip</Button>
        </div>
        {correcting && (
          <CorrectionPanel did={did} pv={pv} busy={busy}
            onApply={(c) => act("correct", c)} />
        )}
      </Card>
      {debug && (
        <div className="space-y-3">
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <DebugBox title="Exact preview payload (data received)" tone="indigo"><JsonBox data={pv} /></DebugBox>
            <DebugBox title="Identification internals" tone="amber"><JsonBox data={pv.identification} /></DebugBox>
          </div>
          <h4 className="text-sm font-semibold text-slate-700">Agents — objects, values, functions, curator (live)</h4>
          <AgentPanels d={dbg} />
        </div>
      )}
    </div>
  );
}

function CorrectionPanel({ did, pv, busy, onApply }) {
  const [label, setLabel] = useState(pv.label_hint || pv.identification.proposed || "");
  const [value, setValue] = useState("");
  return (
    <div className="mt-3 space-y-3 rounded-lg border border-amber-200 bg-amber-50/40 p-3">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-amber-700">
        Correct this item (sticky, human-sourced) + tune learning parameters
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-[11px] text-slate-600">correct class label</label>
          <input value={label} onChange={(e) => setLabel(e.target.value)}
            className="w-44 px-2 py-1 text-sm" />
        </div>
        <div>
          <label className="block text-[11px] text-slate-600">objective value override (optional)</label>
          <input type="number" step="0.01" value={value} placeholder="e.g. 0.95"
            onChange={(e) => setValue(e.target.value)} className="w-32 px-2 py-1 text-sm" />
        </div>
        <Button variant="amber" disabled={busy}
          onClick={() => onApply({ label, ...(value !== "" ? { value: Number(value) } : {}) })}>
          Apply correction (commit item)
        </Button>
      </div>
      <details open>
        <summary className="cursor-pointer text-[12px] font-medium text-slate-700">
          Learning math &amp; weights — edit all parameters
        </summary>
        <div className="mt-2"><ParamEditor compact /></div>
      </details>
    </div>
  );
}

function Unsupervised({ did }) {
  const { refresh } = useDomain();
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async () => { setBusy(true); const r = await api.trainAuto(did, 18); setRes(r); refresh(); setBusy(false); };
  return (
    <Card title="Unsupervised auto-training">
      <p className="text-sm text-slate-500">Auto-ingest the domain's data: identify, learn values/affordances, and let the Curator update memory + vectors — no per-item confirm.</p>
      <Button className="mt-2" disabled={busy} onClick={run}>{busy ? "Running…" : "Run auto-training"}</Button>
      {res && (
        <div className="mt-3 space-y-2">
          <div className="flex gap-2 text-sm">
            <Chip color="indigo">ticks {res.ticks}</Chip>
            <Chip>memory {res.memory_size}</Chip>
            <Chip color={res.novelty_rate < 0.15 ? "green" : "amber"}>novelty {res.novelty_rate}</Chip>
          </div>
          <div className="max-h-48 overflow-auto rounded bg-slate-50 p-2 text-[11px] font-mono text-slate-500">
            {res.progress.map((p) => (
              <div key={p.tick}>t{p.tick} · {p.class} · {p.memory_action} · mem={p.memory_size}</div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}
