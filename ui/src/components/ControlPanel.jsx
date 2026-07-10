import { useState } from "react";
import Card from "./Card.jsx";

export default function ControlPanel({ status, onConfigure, onControl, onToggleReview }) {
  const [domain, setDomain] = useState("synthetic");
  const review = status?.status?.review_mode ?? true;
  const running = status?.status?.running;
  return (
    <Card title="Control">
      <div className="flex flex-wrap items-center gap-2">
        <input value={domain} onChange={(e) => setDomain(e.target.value)}
          placeholder="domain id" className="rounded bg-white px-2 py-1 text-sm" />
        <button onClick={() => onConfigure(domain, review)}
          className="rounded bg-indigo-600 px-3 py-1 text-sm hover:bg-indigo-500">Configure</button>
        <button onClick={() => onControl("start", "train")}
          className="rounded bg-emerald-600 px-3 py-1 text-sm hover:bg-emerald-500">Start training</button>
        <button onClick={() => onControl("step")}
          className="rounded bg-slate-100 px-3 py-1 text-sm hover:bg-slate-200">Step</button>
        <button onClick={() => onControl("pause")}
          className="rounded bg-amber-600 px-3 py-1 text-sm hover:bg-amber-500">Pause</button>
        <button onClick={() => onControl("stop")}
          className="rounded bg-rose-700 px-3 py-1 text-sm hover:bg-rose-600">Stop</button>
        <label className="ml-auto flex items-center gap-2 text-sm">
          <input type="checkbox" checked={review} onChange={(e) => onToggleReview(e.target.checked)} />
          Review mode {review ? "ON" : "OFF (seamless)"}
        </label>
      </div>
      <div className="mt-3 text-xs text-slate-500">
        mode={status?.status?.mode} · tick={status?.ticks_done ?? 0} · memory={status?.memory_size ?? 0}
        · {running ? "running" : "idle"}
      </div>
    </Card>
  );
}
