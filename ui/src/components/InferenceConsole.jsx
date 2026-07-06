import { useState } from "react";
import Card from "./Card.jsx";
export default function InferenceConsole({ onInfer }) {
  const [text, setText] = useState("possible sars_cov_2 signature in sample 7");
  const [result, setResult] = useState(null);
  const run = async () => setResult(await onInfer({ text }));
  return (
    <Card title="Inference console">
      <div className="flex gap-2">
        <input value={text} onChange={(e) => setText(e.target.value)}
          className="flex-1 rounded bg-white px-2 py-1 text-sm" placeholder="describe a problem…" />
        <button onClick={run} className="rounded bg-indigo-600 px-3 py-1 text-sm hover:bg-indigo-500">Infer</button>
      </div>
      {result && (
        <div className={`mt-3 rounded border p-2 text-xs ${result.ood ? "border-rose-700 bg-rose-50" : "border-emerald-700 bg-emerald-50"}`}>
          {result.ood ? (
            <p className="font-semibold text-rose-600">Outside trained domain — no action.</p>
          ) : (
            <p className="font-semibold text-emerald-700">Action: {result.action} (conf {result.confidence})</p>
          )}
          <p className="mt-1 text-slate-500">{result.rationale}</p>
        </div>
      )}
    </Card>
  );
}
