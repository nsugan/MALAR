import { useState } from "react";
import { DomainProvider, useDomain } from "./context/DomainContext.jsx";
import DomainsTab from "./tabs/DomainsTab.jsx";
import ConfigureTab from "./tabs/ConfigureTab.jsx";
import TrainTab from "./tabs/TrainTab.jsx";
import WorldGraphTab from "./tabs/WorldGraphTab.jsx";
import KnowledgeTab from "./tabs/KnowledgeTab.jsx";
import TestInferenceTab from "./tabs/TestInferenceTab.jsx";
import { Chip } from "./components/Card.jsx";
import DebugConsole from "./components/DebugConsole.jsx";
import LLMTab from "./tabs/LLMTab.jsx";
import AgentsTab from "./tabs/AgentsTab.jsx";
import InferencePredictionTab from "./tabs/InferencePredictionTab.jsx";

const TABS = [
  ["domains", "Domains", DomainsTab],
  ["configure", "Configure", ConfigureTab],
  ["train", "Train", TrainTab],
  ["graph", "World Graph", WorldGraphTab],
  ["knowledge", "Knowledge", KnowledgeTab],
  ["test", "Test & Inference", TestInferenceTab],
  ["agents", "Agents", AgentsTab],
  ["llm", "LLM", LLMTab],
  ["debug", "Debug", DebugConsole],
  ["predict", "Inference and Prediction", InferencePredictionTab],
];

function Header() {
  const { domains, activeId, select, mode, review, setReview, debug, setDebug } = useDomain();
  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur border-slate-200">
      <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
        <h1 className="text-lg font-bold tracking-tight">MALAR <span className="text-indigo-400">studio</span></h1>
        <div className="ml-2 flex items-center gap-2">
          <span className="text-xs text-slate-500">domain</span>
          <select value={activeId || ""} onChange={(e) => select(e.target.value)}
            className="rounded bg-white px-2 py-1 text-sm">
            {domains.length === 0 && <option value="">— none —</option>}
            {domains.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
        </div>
        <Chip color={mode === "supervised" ? "indigo" : "green"}>{mode}</Chip>
        <label className="ml-auto flex items-center gap-2 text-xs text-slate-500">
          <input type="checkbox" checked={review} onChange={(e) => setReview(e.target.checked)} />
          Review mode {review ? "ON" : "OFF"}
        </label>
        <label className={`flex items-center gap-2 rounded px-2 py-1 text-xs transition
          ${debug ? "bg-emerald-50 text-emerald-700" : "text-slate-500"}`}>
          <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} />
          Debug
        </label>
      </div>
    </header>
  );
}

function Shell() {
  const [tab, setTab] = useState("domains");
  const Active = TABS.find((t) => t[0] === tab)[2];
  return (
    <div className="min-h-screen">
      <Header />
      <nav className="mx-auto max-w-7xl px-4 pt-4">
        <div className="flex flex-wrap gap-1 rounded-lg border border-slate-200 bg-white p-1">
          {TABS.map(([id, label]) => (
            <button key={id} onClick={() => setTab(id)}
              className={`rounded px-3 py-1.5 text-sm transition ${tab === id
                ? "bg-indigo-600 text-white" : "text-slate-700 hover:bg-slate-100"}`}>
              {label}
            </button>
          ))}
        </div>
      </nav>
      <main className="mx-auto max-w-7xl px-4 py-4"><Active /></main>
    </div>
  );
}

export default function App() {
  const params = new URLSearchParams(window.location.search);
  if (params.get("view") === "graph") {
    const did = params.get("domain");
    return (
      <DomainProvider>
        <GraphWindow did={did} />
      </DomainProvider>
    );
  }
  return <DomainProvider><Shell /></DomainProvider>;
}

function GraphWindow({ did }) {
  const { setActiveId } = useDomain();
  if (did) setActiveId(did);
  return (
    <div className="min-h-screen p-4">
      <h1 className="mb-3 text-lg font-bold">MALAR — World Graph <span className="text-slate-500">({did})</span></h1>
      <WorldGraphTab embedded />
    </div>
  );
}
