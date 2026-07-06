import { useEffect, useState } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import Card, { Button, Chip } from "../components/Card.jsx";

// --- tiny inline visualizations (no external chart dep) -----------------
const BAR = { indigo: "bg-indigo-500", sky: "bg-sky-500", emerald: "bg-emerald-500" };
function Bars({ dist, color = "indigo", max = 8 }) {
  const items = Object.entries(dist || {}).sort((a, b) => b[1] - a[1]).slice(0, max);
  if (!items.length) return <p className="text-[11px] text-slate-400">—</p>;
  return (
    <div className="space-y-1">
      {items.map(([k, v]) => (
        <div key={k} className="flex items-center gap-2">
          <span className="w-28 truncate text-[11px] text-slate-600">{k}</span>
          <div className="h-3 flex-1 rounded bg-slate-100">
            <div className={`h-3 rounded ${BAR[color] || BAR.indigo}`} style={{ width: `${Math.max(2, v * 100)}%` }} />
          </div>
          <span className="w-10 text-right text-[11px] tabular-nums text-slate-500">{(v * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

function Spark({ values }) {
  if (!values || !values.length) return null;
  const w = 280, h = 44, lo = Math.min(...values), hi = Math.max(...values);
  const sx = w / (values.length - 1 || 1), sy = hi > lo ? (h - 6) / (hi - lo) : 0;
  const pts = values.map((v, i) => `${(i * sx).toFixed(1)},${(h - 3 - (v - lo) * sy).toFixed(1)}`).join(" ");
  return (
    <svg width={w} height={h} className="rounded bg-slate-50">
      <polyline points={pts} fill="none" stroke="#6366f1" strokeWidth="1.2" />
    </svg>
  );
}

export default function InferencePredictionTab() {
  const { activeId, debug } = useDomain();
  const [text, setText] = useState("");
  const [file, setFile] = useState("");
  const [priorMode, setPriorMode] = useState("prevalence");
  const [nSamples, setNSamples] = useState(2000);
  const [fullMcmc, setFullMcmc] = useState(false);
  const [crossDomain, setCrossDomain] = useState(false);
  const [wTopo, setWTopo] = useState(1), [wSpec, setWSpec] = useState(1), [wGraph, setWGraph] = useState(1);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [agents, setAgents] = useState(null);
  const [fbMsg, setFbMsg] = useState("");

  useEffect(() => { if (activeId) api.predictAgents(activeId).then(setAgents).catch(() => {}); }, [activeId]);
  if (!activeId) return <p className="text-sm text-slate-500">Select or create a domain first.</p>;

  const run = async () => {
    setBusy(true);
    try {
      const r = await api.predict(activeId, {
        text: text || null, file: file || null, prior_mode: priorMode,
        weights: { topo: Number(wTopo), spectral: Number(wSpec), graph: Number(wGraph) },
        n_samples: Number(nSamples), mcmc_method: fullMcmc ? "nuts" : "mc",
        cross_domain: crossDomain, narrate: true,
      });
      setRes(r);
      setAgents(await api.predictAgents(activeId));
    } catch (e) { setRes({ error: String(e) }); }
    setBusy(false);
  };

  const sendFeedback = async (action, reward) => {
    try { const r = await api.predictFeedback(activeId, { action, reward });
      setFbMsg(r.ok ? `recorded ${reward > 0 ? "✓" : "✗"} for '${action}' (updates ${r.updates})` : (r.reason || "failed"));
      setAgents(await api.predictAgents(activeId));
    } catch (e) { setFbMsg(String(e)); }
  };

  const b = res?.bayes;
  const m = res?.mcmc;

  return (
    <div className="space-y-4">
      <Card title="Inference & Prediction — Bayesian fusion + MCMC forecast">
        <p className="text-xs text-slate-500">
          Each modality runs a Bayesian classifier fit over the whole trained corpus; the
          Inference Management agent fuses them (product-of-experts) and runs an MCMC
          posterior-predictive simulation to forecast the outcome with credible intervals.
        </p>
        <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
          <div>
            <label className="text-xs text-slate-500">Query text</label>
            <textarea value={text} onChange={(e) => setText(e.target.value)}
              placeholder="e.g. possible coronavirus signature in sample 3"
              className="mt-1 h-20 w-full rounded bg-white px-2 py-1.5 text-sm" />
            <label className="mt-2 block text-xs text-slate-500">…or a server-visible spectra file</label>
            <input value={file} onChange={(e) => setFile(e.target.value)}
              placeholder="/app/Raman_Virus/RSV/sample_01.txt"
              className="mt-1 w-full rounded bg-white px-2 py-1.5 text-sm" />
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-24 text-xs text-slate-500">Prior</span>
              <select value={priorMode} onChange={(e) => setPriorMode(e.target.value)}
                className="rounded bg-white px-2 py-1 text-sm">
                <option value="prevalence">prevalence</option>
                <option value="uniform">uniform</option>
              </select>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-24 text-xs text-slate-500">Modality wts</span>
              {[["topo", wTopo, setWTopo], ["spec", wSpec, setWSpec], ["graph", wGraph, setWGraph]].map(([n, v, set]) => (
                <label key={n} className="flex items-center gap-1 text-[11px] text-slate-500">{n}
                  <input type="number" step="0.5" min="0" value={v} onChange={(e) => set(e.target.value)}
                    className="w-12 rounded bg-white px-1 py-0.5 text-xs" /></label>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <span className="w-24 text-xs text-slate-500">MCMC samples</span>
              <input type="number" step="500" min="100" value={nSamples} onChange={(e) => setNSamples(e.target.value)}
                className="w-24 rounded bg-white px-2 py-1 text-sm" />
            </div>
            <label className="flex items-center gap-2 text-xs text-slate-600">
              <input type="checkbox" checked={fullMcmc} onChange={(e) => setFullMcmc(e.target.checked)} />
              Full MCMC (emcee/PyMC NUTS) — needs the <code>[ml]</code> extra
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-600">
              <input type="checkbox" checked={crossDomain} onChange={(e) => setCrossDomain(e.target.checked)} />
              Cross-domain referencing (opt-in · second pass)
            </label>
          </div>
        </div>
        <div className="mt-3">
          <Button disabled={busy} onClick={run}>{busy ? "Predicting…" : "Run prediction"}</Button>
        </div>
      </Card>

      {res?.error && <Card title="Error"><p className="text-sm text-rose-600">{res.error}</p></Card>}

      {res && !res.error && (
        <Card title="Verdict" right={res.ood ? <Chip color="amber">outside trained domain</Chip>
          : <Chip color="green">in-distribution</Chip>}>
          <p className="text-sm text-slate-800">{res.verdict}</p>
          <div className="mt-2 flex flex-wrap gap-2 text-xs">
            {res.predicted_action && <Chip color="indigo">action: {res.predicted_action}</Chip>}
            {res.action_confidence != null && <Chip>P(action) {(res.action_confidence * 100).toFixed(0)}%</Chip>}
            {b?.top && <Chip>class {b.top} · {(b.top_p * 100).toFixed(0)}%</Chip>}
            <Chip>match conf {((res.match_confidence || 0) * 100).toFixed(0)}%</Chip>
            <Chip>corpus n={b?.n_corpus ?? "—"}</Chip>
          </div>
        </Card>
      )}

      {b && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card title="Per-modality posteriors">
            {["topo", "spectral", "graph"].map((mm) => b.per_modality?.[mm] && (
              <div key={mm} className="mb-3">
                <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">{mm}
                  <span className="ml-2 text-slate-400">evidence {((b.evidence?.[mm] || 0) * 100).toFixed(0)}%</span></div>
                <Bars dist={b.per_modality[mm]} color="sky" />
              </div>
            ))}
          </Card>
          <Card title="Fused posterior (product of experts)" right={<Chip>entropy {(b.entropy || 0).toFixed(2)}</Chip>}>
            <Bars dist={b.fused} color="indigo" />
          </Card>
        </div>
      )}

      {m && (
        <Card title="MCMC posterior-predictive forecast" right={<Chip>{m.method} · {m.n_samples} samples</Chip>}>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">Predicted action</div>
              <Bars dist={m.action_predictive} color="emerald" />
              <div className="mt-3 mb-1 text-[10px] uppercase tracking-wider text-slate-500">
                Trace · {m.trace_objective}</div>
              <Spark values={m.trace} />
            </div>
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">Objective forecast (95% CI)</div>
              <table className="w-full text-xs">
                <thead><tr className="text-slate-400"><th className="text-left">objective</th>
                  <th className="text-right">mean</th><th className="text-right">95% CI</th></tr></thead>
                <tbody>
                  {Object.entries(m.value_summary || {}).map(([k, v]) => (
                    <tr key={k} className="border-t border-slate-100">
                      <td className="py-1 text-slate-700">{k}</td>
                      <td className="py-1 text-right tabular-nums">{v.mean.toFixed(3)}</td>
                      <td className="py-1 text-right tabular-nums text-slate-500">
                        [{v.ci_low.toFixed(2)}, {v.ci_high.toFixed(2)}]</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      )}

      {res?.cross_domain && (
        <Card title="Cross-domain references" right={<Chip color={res.cross_domain.enabled ? "green" : "slate"}>
          {res.cross_domain.enabled ? "on" : "opt-in"}</Chip>}>
          {(res.cross_domain.matches || []).length > 0 ? (
            <table className="w-full text-xs">
              <thead><tr className="text-slate-400"><th className="text-left">domain</th>
                <th className="text-left">class</th><th className="text-right">similarity</th>
                <th className="text-right">weight</th></tr></thead>
              <tbody>
                {res.cross_domain.matches.map((m, i) => (
                  <tr key={i} className="border-t border-slate-100">
                    <td className="py-1 text-slate-700">{m.domain}</td>
                    <td className="py-1 text-slate-600">{m.class}</td>
                    <td className="py-1 text-right tabular-nums">{(m.score * 100).toFixed(1)}%</td>
                    <td className="py-1 text-right tabular-nums text-amber-600">×{m.weight}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p className="text-[11px] text-slate-500">
            {res.cross_domain.enabled ? "no matches in other loaded domains" : res.cross_domain.note}</p>}
        </Card>
      )}

      {res?.diffusion && !res.diffusion.error && (
        <Card title="Diffusion field (image/video)" right={<Chip>stage {res.diffusion.stage} · {res.diffusion.method || "—"}</Chip>}>
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">Diffusion class vote</div>
              <Bars dist={res.diffusion.diffusion_vote} color="sky" />
              {res.diffusion.top && <div className="mt-1 text-[11px] text-slate-600">top region class: <b>{res.diffusion.top}</b></div>}
            </div>
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
                Per-region heat ({(res.diffusion.heatmap || []).length} regions)</div>
              <Spark values={res.diffusion.heatmap} />
              <p className="mt-1 text-[10px] text-slate-400">{res.diffusion.note}</p>
            </div>
          </div>
        </Card>
      )}

      {res?.rl && (
        <Card title="RL policy (contextual bandit · Thompson)" right={<Chip color="indigo">{res.rl.updates} updates</Chip>}>
          <div className="mb-2 text-sm text-slate-700">sampled action: <b>{res.rl.action}</b></div>
          <Bars dist={res.rl.mean_reward} color="emerald" />
          {res.predicted_action && (
            <div className="mt-3 flex items-center gap-2">
              <span className="text-[11px] text-slate-500">Was action ‘{res.predicted_action}’ correct?</span>
              <Button variant="green" onClick={() => sendFeedback(res.predicted_action, 1)}>✓ reward</Button>
              <Button variant="ghost" onClick={() => sendFeedback(res.predicted_action, 0)}>✗ no</Button>
              {fbMsg && <span className="text-[11px] text-slate-500">{fbMsg}</span>}
            </div>
          )}
        </Card>
      )}

      {(debug || true) && agents && (
        <Card title="Inference agents" right={<Chip>{(agents.agents || []).filter((a) => a.enabled).length} active</Chip>}>
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
            {(agents.agents || []).map((a) => (
              <div key={a.id} className={`rounded-lg border p-2 ${a.enabled
                ? "border-emerald-200 bg-emerald-50" : "border-slate-200 bg-slate-50 opacity-70"}`}>
                <div className="text-xs font-medium text-slate-700">{a.name}</div>
                <div className="text-[10px] text-slate-500">{a.modality}</div>
                {a.last && <div className="mt-1 text-[10px] text-slate-600">
                  {a.last.top ? `${a.last.top} · ${((a.last.p || 0) * 100).toFixed(0)}%` :
                   (a.last.verdict || "")}</div>}
                {!a.enabled && <div className="mt-1 text-[10px] text-amber-600">{a.note}</div>}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
