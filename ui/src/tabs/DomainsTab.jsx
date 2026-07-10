import { useState } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import Card, { Chip, Button } from "../components/Card.jsx";

export default function DomainsTab() {
  const { domains, activeId, select, refresh } = useDomain();
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [purge, setPurge] = useState(true);
  const [busy, setBusy] = useState(false);

  const create = async () => {
    if (!name.trim()) return;
    // All domains are generic/synthetic; what the data actually is gets described here
    // and refined in the Configure tab. No instrument/modality is baked in.
    await api.createDomain({ name, description: desc, adapter_type: "synthetic" });
    setName(""); setDesc(""); refresh();
  };
  const mem = purge ? " AND its learned memory (Qdrant + Neo4j)" : " (memory kept)";
  const reset = async (id) => { if (confirm(`Reset ${id}? Wipes artifacts${mem}. Config kept.`)) { await api.resetDomain(id, purge); refresh(); } };
  const remove = async (id) => { if (confirm(`Delete ${id}? Removes the domain${mem}.`)) { await api.deleteDomain(id, purge); refresh(); } };
  const wipeAll = async () => {
    if (!confirm("FRESH START: drop EVERY domain's Qdrant collection and clear ALL of Neo4j. This cannot be undone. Continue?")) return;
    setBusy(true);
    try { const r = await api.wipeMemory(); alert(`Cleared. Neo4j nodes deleted: ${r.neo4j ?? 0}; domains purged: ${(r.domains || []).length}.`); }
    catch (e) { alert("Wipe failed: " + e); }
    setBusy(false); refresh();
  };

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <Card title="New domain" className="lg:col-span-1">
        <div className="space-y-2">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Domain name"
            className="w-full rounded bg-white px-2 py-1.5 text-sm" />
          <textarea value={desc} onChange={(e) => setDesc(e.target.value)}
            placeholder="What is this data? (subject, labels, source) — details go in Configure"
            className="h-24 w-full rounded bg-white px-2 py-1.5 text-sm" />
          <Button onClick={create} className="w-full">Add domain</Button>
        </div>
      </Card>

      <Card title={`Domains (${domains.length})`} className="lg:col-span-2"
        right={
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
              <input type="checkbox" checked={purge} onChange={(e) => setPurge(e.target.checked)} />
              also wipe Qdrant + Neo4j
            </label>
            <Button variant="red" disabled={busy} onClick={wipeAll}>{busy ? "Wiping…" : "Fresh start (wipe all memory)"}</Button>
          </div>
        }>
        {domains.length === 0 ? (
          <p className="text-sm text-slate-500">No domains yet — create one to begin.</p>
        ) : (
          <div className="space-y-2">
            {domains.map((d) => (
              <div key={d.id} className={`flex items-center justify-between rounded-lg border px-3 py-2
                ${d.id === activeId ? "border-indigo-500 bg-indigo-50" : "border-slate-200 bg-slate-50"}`}>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-slate-900">{d.name}</span>
                    <Chip color="indigo">{d.adapter_type}</Chip>
                    {d.id === activeId && <Chip color="green">active</Chip>}
                  </div>
                  <div className="mt-0.5 flex gap-3 text-[11px] text-slate-500">
                    <span>{d.objects_learned} objects</span>
                    <span>{d.memory_size} memories</span>
                    <span className="text-slate-500">{d.id}</span>
                  </div>
                </div>
                <div className="flex gap-1.5">
                  {d.id !== activeId && <Button variant="ghost" onClick={() => select(d.id)}>Select</Button>}
                  <Button variant="amber" onClick={() => reset(d.id)}>Reset</Button>
                  <Button variant="red" onClick={() => remove(d.id)}>Delete</Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
