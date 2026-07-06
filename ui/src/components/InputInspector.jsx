import { useEffect, useState } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { DebugBox, KV, JsonBox } from "./DebugConsole.jsx";
import Heatmap from "./Heatmap.jsx";
import GraphView from "./GraphView.jsx";
import TopologyView from "./TopologyView.jsx";

const LINE_COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#06b6d4", "#a855f7", "#84cc16", "#ec4899"];

export default function InputInspector() {
  const { activeId } = useDomain();
  const [items, setItems] = useState([]);
  const [idx, setIdx] = useState(0);
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [src, setSrc] = useState(null);

  useEffect(() => {
    if (!activeId) return;
    api.inputs(activeId).then((r) => { setItems(r.items || []); setIdx(0); setSrc(r); });
  }, [activeId]);

  const inspect = async (i) => {
    setBusy(true); setIdx(i);
    try { setData(await api.inspect(activeId, i)); } catch (e) { setData({ error: String(e) }); }
    setBusy(false);
  };
  useEffect(() => { if (activeId && items.length) inspect(0); /* eslint-disable-next-line */ }, [items, activeId]);

  if (!activeId) return null;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-slate-200 bg-white p-3">
        <span className="text-xs font-semibold text-slate-600">Input file</span>
        <select value={idx} onChange={(e) => inspect(Number(e.target.value))}
          className="min-w-[260px] px-2 py-1 text-sm">
          {items.map((it) => (
            <option key={it.index} value={it.index}>
              #{it.index} · {it.item_id} · [{it.label}] · {it.shape?.join("×")}
            </option>
          ))}
        </select>
        {busy && <span className="text-xs text-slate-400">inspecting…</span>}
        <span className="ml-auto text-[11px] text-slate-400">{items.length} files · {src?.source}</span>
      </div>

      {!data ? <p className="text-sm text-slate-500">Choose a file to inspect.</p> :
       data.error ? <p className="text-sm text-rose-600">{data.error}</p> : (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <DebugBox title="① File — what was received" tone="indigo">
            <KV obj={data.file} />
            <div className="mt-2"><KV obj={data.raw.stats} /></div>
          </DebugBox>

          <DebugBox title="② Display — spectra (samples + mean)" tone="indigo">
            <SpectraPlot spectra={data.spectra} />
          </DebugBox>

          <DebugBox title="③ Raw value matrix (rows × bands)">
            <Heatmap matrix={data.raw.matrix} label={`${data.raw.rows_shown}×${data.raw.cols_shown} values`} />
          </DebugBox>

          <DebugBox title="④ Per-row norms">
            <BarRow values={data.raw.per_row_norm} />
            <details className="mt-2"><summary className="cursor-pointer text-[11px] text-slate-500">raw values (first rows)</summary>
              <JsonBox data={data.raw.matrix} /></details>
          </DebugBox>

          <DebugBox title="⑤ Structure — graph" tone="green">
            <GraphView data={data.structure.graph} height={200} />
            <div className="mt-1 text-[11px] text-slate-500">
              {data.structure.n_nodes} nodes · {data.structure.n_edges} edges
            </div>
          </DebugBox>

          <DebugBox title="⑥ Structure — adjacency matrix" tone="green">
            <Heatmap matrix={data.structure.adjacency} cellW={11} cellH={11} maxCols={32}
              label="weighted adjacency A" />
            <details className="mt-2"><summary className="cursor-pointer text-[11px] text-slate-500">degrees</summary>
              <JsonBox data={data.structure.degrees} /></details>
          </DebugBox>

          <DebugBox title="⑦ Topology — persistence (H0/H1/H2)" tone="amber">
            <TopologyView diagram={data.topology.h1} summary={data.topology.summary} />
            <div className="mt-2 grid grid-cols-3 gap-1 text-[11px]">
              {["h0", "h1", "h2"].map((h) => (
                <div key={h} className="rounded bg-slate-50 p-1">
                  <div className="text-slate-500">{h.toUpperCase()}</div>
                  <div className="font-mono text-slate-700">{data.topology[h].length} pts</div>
                </div>
              ))}
            </div>
            <details className="mt-2"><summary className="cursor-pointer text-[11px] text-slate-500">diagrams (birth, death)</summary>
              <JsonBox data={{ H0: data.topology.h0, H1: data.topology.h1, H2: data.topology.h2 }} /></details>
          </DebugBox>

          <DebugBox title="⑧ Hyperspectral — encode / reconstruct" tone="amber">
            <ReconPlot mean={data.spectral.mean_spectrum} recon={data.spectral.reconstructed_mean} />
            <div className="mt-1 text-[11px] text-slate-500">recon error: <b className="text-slate-700">{data.spectral.recon_error}</b></div>
            <div className="mt-1 text-[10px] uppercase tracking-wider text-slate-400">latent codes (per node)</div>
            <Heatmap matrix={data.spectral.latent} cellW={9} cellH={9} maxCols={16} label="Z (n × latent)" />
          </DebugBox>

          <DebugBox title="⑨ Graph embedding (g)">
            <BarRow values={data.graph_emb.vector} />
            <div className="mt-1"><KV obj={data.graph_emb.summary} /></div>
          </DebugBox>

          <DebugBox title="⑩ Process — identification & decision" tone="indigo">
            <KV obj={data.process.identification} />
            <div className="mt-2 text-[10px] uppercase tracking-wider text-slate-400">objective values R</div>
            <KV obj={data.process.r_values} />
            <div className="mt-2 text-[10px] uppercase tracking-wider text-slate-400">affordances F</div>
            <div className="flex flex-wrap gap-1">{(data.process.affordances || []).map((a) =>
              <span key={a} className="rounded bg-emerald-50 px-2 py-0.5 text-[10px] text-emerald-700 ring-1 ring-emerald-200">{a}</span>)}</div>
            <div className="mt-2 text-[10px] uppercase tracking-wider text-slate-400">novelty vs nearest memory</div>
            {data.process.novelty_vs_nearest
              ? <KV obj={data.process.novelty_vs_nearest} />
              : <span className="text-[11px] text-slate-400">first of its kind (no neighbor yet)</span>}
          </DebugBox>
        </div>
      )}
    </div>
  );
}

function SpectraPlot({ spectra }) {
  const n = spectra.mean.length;
  const rows = [];
  for (let x = 0; x < n; x++) {
    const row = { x, mean: spectra.mean[x] };
    spectra.samples.forEach((s, si) => { row[`s${si}`] = s[x]; });
    rows.push(row);
  }
  return (
    <ResponsiveContainer width="100%" height={170}>
      <LineChart data={rows} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
        <XAxis dataKey="x" tick={{ fontSize: 9, fill: "#475569" }} />
        <YAxis tick={{ fontSize: 9, fill: "#475569" }} />
        <Tooltip contentStyle={{ fontSize: 11, border: "1px solid #e2e8f0" }} />
        {spectra.samples.map((_, si) => (
          <Line key={si} type="monotone" dataKey={`s${si}`} stroke={LINE_COLORS[si % LINE_COLORS.length]}
            dot={false} strokeWidth={0.8} opacity={0.5} />
        ))}
        <Line type="monotone" dataKey="mean" stroke="#0f172a" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function ReconPlot({ mean, recon }) {
  const rows = mean.map((y, x) => ({ x, original: y, reconstructed: recon[x] }));
  return (
    <ResponsiveContainer width="100%" height={150}>
      <LineChart data={rows} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
        <XAxis dataKey="x" tick={{ fontSize: 9, fill: "#475569" }} />
        <YAxis tick={{ fontSize: 9, fill: "#475569" }} />
        <Tooltip contentStyle={{ fontSize: 11, border: "1px solid #e2e8f0" }} />
        <Legend wrapperStyle={{ fontSize: 10 }} />
        <Line type="monotone" dataKey="original" stroke="#6366f1" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="reconstructed" stroke="#f59e0b" dot={false} strokeWidth={1.2} strokeDasharray="3 2" />
      </LineChart>
    </ResponsiveContainer>
  );
}

function BarRow({ values }) {
  const max = Math.max(1e-9, ...values.map(Math.abs));
  return (
    <div className="flex h-12 items-end gap-[1px]">
      {values.slice(0, 64).map((v, i) => (
        <div key={i} title={`${i}: ${v}`} className="flex-1 rounded-t bg-indigo-400"
          style={{ height: `${(Math.abs(v) / max) * 100}%`, minWidth: 2 }} />
      ))}
    </div>
  );
}
