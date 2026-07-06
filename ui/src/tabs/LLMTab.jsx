import { useEffect, useState, useCallback } from "react";
import { api } from "../lib/api.js";
import Card, { Button, Chip } from "../components/Card.jsx";

export default function LLMTab() {
  const [status, setStatus] = useState(null);
  const [log, setLog] = useState([]);
  const [auto, setAuto] = useState(true);

  const loadLog = useCallback(() => { api.llmLog(200).then((r) => setLog(r.calls || [])); }, []);
  useEffect(() => { api.llmStatus().then(setStatus); loadLog(); }, [loadLog]);
  useEffect(() => {
    if (!auto) return;
    const id = setInterval(loadLog, 2500);
    return () => clearInterval(id);
  }, [auto, loadLog]);

  return (
    <div className="space-y-4">
      <GatewayStatus status={status} onRefresh={() => api.llmStatus().then(setStatus)} />
      <ProviderRouting />
      <Console aliases={status?.aliases} onSent={loadLog} />
      <Card title={`LLM call log (${log.length})`}
        right={
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-1 text-xs text-slate-500">
              <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> auto
            </label>
            <Button variant="ghost" onClick={loadLog}>Refresh</Button>
            <Button variant="red" onClick={() => api.llmClear().then(loadLog)}>Clear</Button>
          </div>
        }>
        {log.length === 0 ? (
          <p className="text-sm text-slate-500">
            No LLM calls yet. MALAR is metric-first, so the LLM is only queried for cold-start /
            uncertain identifications, text inference parsing, and explanations — or use the console
            above to send one manually.
          </p>
        ) : (
          <div className="space-y-2">{log.map((c) => <LogEntry key={c.id} c={c} />)}</div>
        )}
      </Card>
    </div>
  );
}

function GatewayStatus({ status, onRefresh }) {
  if (!status) return <Card title="Gateway"><p className="text-sm text-slate-500">Checking…</p></Card>;
  return (
    <Card title="LLM gateway (LiteLLM)" right={<Button variant="ghost" onClick={onRefresh}>Recheck</Button>}>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Chip color={status.reachable ? "green" : "red"}>{status.reachable ? "reachable" : "unreachable"}</Chip>
        <span className="font-mono text-xs text-slate-500">{status.base_url}</span>
        {status.error && <span className="text-xs text-rose-600">{status.error}</span>}
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {(status.models || []).map((m) => <Chip key={m} color="indigo">{m}</Chip>)}
        {(!status.models || status.models.length === 0) &&
          <span className="text-xs text-slate-400">no models reported</span>}
      </div>
      <div className="mt-2 text-[11px] text-slate-500">
        aliases — reasoner: <b>{status.aliases?.reasoner}</b> · fast: <b>{status.aliases?.fast}</b> · vision: <b>{status.aliases?.vision}</b>
      </div>
    </Card>
  );
}

function Console({ aliases, onSent }) {
  const [alias, setAlias] = useState("malar-reasoner");
  const [system, setSystem] = useState("");
  const [prompt, setPrompt] = useState("Reply with the single word: ok");
  const [resp, setResp] = useState(null);
  const [busy, setBusy] = useState(false);

  const send = async () => {
    setBusy(true); setResp(null);
    const r = await api.llmComplete({ alias, prompt, system: system || null });
    setResp(r); setBusy(false); onSent && onSent();
  };
  const opts = [aliases?.reasoner, aliases?.fast, aliases?.vision, "malar-frontier"].filter(Boolean);

  return (
    <Card title="LLM console — send a prompt, see the output">
      <div className="grid grid-cols-1 gap-2 lg:grid-cols-[160px_1fr]">
        <select value={alias} onChange={(e) => setAlias(e.target.value)} className="px-2 py-1.5 text-sm">
          {(opts.length ? opts : ["malar-reasoner", "malar-fast", "malar-vision"]).map((a) =>
            <option key={a} value={a}>{a}</option>)}
        </select>
        <input value={system} onChange={(e) => setSystem(e.target.value)}
          placeholder="system prompt (optional)" className="px-2 py-1.5 text-sm" />
      </div>
      <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)}
        className="mt-2 h-24 w-full px-2 py-1.5 text-sm font-mono" placeholder="your prompt…" />
      <div className="mt-2 flex items-center gap-2">
        <Button disabled={busy} onClick={send}>{busy ? "Sending…" : "Send to LLM"}</Button>
        <span className="text-[11px] text-slate-400">routes through the gateway by alias; appears in the log below</span>
      </div>
      {resp && (
        <div className={`mt-3 rounded-lg border p-2 text-sm ${resp.ok
          ? "border-emerald-200 bg-emerald-50" : "border-rose-200 bg-rose-50"}`}>
          {resp.ok
            ? <pre className="whitespace-pre-wrap font-mono text-[12px] text-slate-800">{resp.response}</pre>
            : <span className="text-rose-700">{resp.error}</span>}
        </div>
      )}
    </Card>
  );
}

function LogEntry({ c }) {
  const [open, setOpen] = useState(false);
  const t = new Date((c.ts || 0) * 1000).toLocaleTimeString();
  return (
    <div className={`rounded-lg border p-2 ${c.ok ? "border-slate-200 bg-white" : "border-rose-200 bg-rose-50"}`}>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Chip color="indigo">{c.alias}</Chip>
        <Chip color={c.source === "manual" ? "amber" : "slate"}>{c.source}</Chip>
        <Chip color={c.ok ? "green" : "red"}>{c.ok ? "ok" : "error"}</Chip>
        <span className="text-slate-400">{c.latency_ms} ms</span>
        <span className="text-slate-400">{t}</span>
        <button onClick={() => setOpen((v) => !v)} className="ml-auto text-indigo-600">{open ? "hide" : "details"}</button>
      </div>
      <div className="mt-1 truncate text-[12px] text-slate-600"><b>Q:</b> {c.prompt}</div>
      {(c.response || c.error) && (
        <div className="truncate text-[12px] text-slate-700"><b>A:</b> {c.response || c.error}</div>
      )}
      {open && (
        <div className="mt-2 space-y-1">
          {c.system && <Box label="system" text={c.system} />}
          <Box label="prompt (query)" text={c.prompt} />
          <Box label={c.ok ? "response (output)" : "error"} text={c.response || c.error} tone={c.ok ? "ok" : "err"} />
        </div>
      )}
    </div>
  );
}
function ProviderRouting() {
  const [r, setR] = useState(null);
  const [edit, setEdit] = useState({});
  const [saved, setSaved] = useState(false);
  const load = () => api.llmRoute().then((d) => { setR(d); setEdit({}); });
  useEffect(load, []);
  if (!r) return null;

  const options = Array.from(new Set([
    ...(r.available || []),
    ...(r.frontier_aliases || []),
    r.defaults.reasoner, r.defaults.fast, r.defaults.vision,
  ].filter(Boolean)));
  const cur = (role) => edit[role] ?? (r.route[role] || "");
  const apply = async () => {
    const body = {};
    ["reasoner", "fast", "vision"].forEach((role) => { if (edit[role] !== undefined) body[role] = edit[role]; });
    await api.llmSetRoute(body); setSaved(true); setTimeout(() => setSaved(false), 2000); load();
  };
  const isFrontier = (v) => (r.frontier_aliases || []).includes(v);
  const anyFrontier = ["reasoner", "fast", "vision"].some((role) => isFrontier(cur(role)));

  return (
    <Card title="Model routing — which provider the agents use"
      right={saved && <Chip color="green">applied</Chip>}>
      <p className="text-xs text-slate-500">
        Choose the alias for each agent role. Local <b>malar-reasoner/fast/vision</b> (Gemma, no egress)
        is default. Frontier aliases route to <b>Claude / ChatGPT / DeepSeek</b> via API
        (requires the key in <code>.env</code> and <b>sends data off-box</b>).
      </p>
      <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-3">
        {["reasoner", "fast", "vision"].map((role) => (
          <div key={role}>
            <label className="text-[11px] text-slate-600">{role}</label>
            <select value={cur(role)} onChange={(e) => setEdit((x) => ({ ...x, [role]: e.target.value }))}
              className="w-full px-2 py-1.5 text-sm">
              <option value="">(default: {r.defaults[role]})</option>
              {options.map((o) => <option key={o} value={o}>{o}{isFrontier(o) ? "  ⚠ frontier" : ""}</option>)}
            </select>
          </div>
        ))}
      </div>
      <div className="mt-2 flex items-center gap-2">
        <Button onClick={apply}>Apply routing</Button>
        {anyFrontier && <Chip color="amber">⚠ frontier selected — data leaves the box</Chip>}
      </div>
    </Card>
  );
}

function Box({ label, text, tone }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-slate-400">{label}</div>
      <pre className={`max-h-48 overflow-auto whitespace-pre-wrap rounded border p-2 text-[11px] font-mono
        ${tone === "err" ? "border-rose-200 bg-rose-50 text-rose-700"
          : "border-slate-200 bg-slate-50 text-slate-700"}`}>{text}</pre>
    </div>
  );
}
