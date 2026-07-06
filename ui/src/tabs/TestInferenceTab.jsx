import { useState } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import Card, { Button, Chip } from "../components/Card.jsx";
import { DistributionChart } from "../components/Charts.jsx";

export default function TestInferenceTab() {
  const { activeId } = useDomain();
  const [path, setPath] = useState("");
  const [run, setRun] = useState(null);
  const [results, setResults] = useState(null);
  const [cmp, setCmp] = useState(null);
  const [busy, setBusy] = useState(false);

  if (!activeId) return <p className="text-sm text-slate-500">Select or create a domain first.</p>;

  const runInfer = async () => {
    setBusy(true);
    const r = await api.inferFolder(activeId, path);
    setRun(r);
    setResults(await api.results(activeId, r.run_id));
    setCmp(await api.compare(activeId, r.run_id));
    setBusy(false);
  };

  return (
    <div className="space-y-4">
      <Card title="Batch inference over a test folder">
        <div className="flex gap-2">
          <input value={path} onChange={(e) => setPath(e.target.value)}
            placeholder="server-visible test folder path (blank = built-in synthetic test set)"
            className="flex-1 rounded bg-white px-2 py-1.5 text-sm" />
          <Button disabled={busy} onClick={runInfer}>{busy ? "Running…" : "Run inference"}</Button>
        </div>
        {run && <div className="mt-2 text-xs text-slate-500">run <span className="font-mono">{run.run_id}</span> · {run.n_items} items · saved to data/domains/{activeId}/results/</div>}
      </Card>

      {results && (
        <Card title={`Results (${results.rows.length})`}>
          <div className="max-h-80 overflow-auto">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-slate-100 text-slate-500">
                <tr><th className="p-1.5">input</th><th>hint</th><th>identified</th><th>action</th><th>confidence</th><th>OOD</th></tr>
              </thead>
              <tbody>
                {results.rows.map((r, i) => (
                  <tr key={i} className="border-t border-slate-200">
                    <td className="p-1.5 font-mono text-slate-700">{r.input_id}</td>
                    <td className="text-slate-500">{r.label_hint}</td>
                    <td className="text-slate-800">{r.identified_object || "—"}</td>
                    <td>{r.action ? <Chip color="green">{r.action}</Chip> : <span className="text-slate-400">none</span>}</td>
                    <td className="text-slate-700">{r.confidence ?? "—"}</td>
                    <td>{r.ood ? <Chip color="red">OOD</Chip> : <Chip color="green">in</Chip>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {cmp && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card title="Test class distribution"><DistributionChart dist={cmp.class_distribution} /></Card>
          <Card title="Comparison with training">
            <div className="space-y-2 text-sm">
              <Row k="OOD rate" v={cmp.ood_rate} />
              <Row k="items" v={cmp.n_items} />
              <div><div className="text-[10px] uppercase tracking-wider text-slate-500">trained classes</div>
                <div className="mt-1 flex flex-wrap gap-1">{cmp.trained_classes.map((c) => <Chip key={c}>{c}</Chip>)}</div></div>
              <div><div className="text-[10px] uppercase tracking-wider text-slate-500">overlap (test ∩ trained)</div>
                <div className="mt-1 flex flex-wrap gap-1">{cmp.overlap.map((c) => <Chip key={c} color="green">{c}</Chip>)}</div></div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
const Row = ({ k, v }) => (
  <div className="flex justify-between"><span className="text-slate-500">{k}</span><span className="text-slate-800">{v}</span></div>
);
