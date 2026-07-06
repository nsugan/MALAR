import { useState } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import Card, { Chip, Button } from "../components/Card.jsx";

export default function DomainsTab() {
  const { domains, activeId, select, refresh } = useDomain();
  const [name, setName] = useState("");
  const [adapter, setAdapter] = useState("raman");
  const [desc, setDesc] = useState("");

  const create = async () => {
    if (!name.trim()) return;
    await api.createDomain({ name, description: desc, adapter_type: adapter });
    setName(""); setDesc(""); refresh();
  };
  const reset = async (id) => { if (confirm(`Reset ${id}? This wipes its memory but keeps config.`)) { await api.resetDomain(id); refresh(); } };
  const remove = async (id) => { if (confirm(`Delete ${id}? Removes everything.`)) { await api.deleteDomain(id); refresh(); } };

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <Card title="New domain" className="lg:col-span-1">
        <div className="space-y-2">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Domain name"
            className="w-full rounded bg-white px-2 py-1.5 text-sm" />
          <select value={adapter} onChange={(e) => setAdapter(e.target.value)}
            className="w-full rounded bg-white px-2 py-1.5 text-sm">
            <option value="raman">raman (hyperspectral)</option>
            <option value="sensor">sensor (timeseries)</option>
            <option value="synthetic">synthetic</option>
          </select>
          <textarea value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="Short description"
            className="h-20 w-full rounded bg-white px-2 py-1.5 text-sm" />
          <Button onClick={create} className="w-full">Add domain</Button>
        </div>
      </Card>

      <Card title={`Domains (${domains.length})`} className="lg:col-span-2">
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
