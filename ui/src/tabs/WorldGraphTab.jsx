import { useEffect, useState, useCallback } from "react";
import { useDomain } from "../context/DomainContext.jsx";
import { api } from "../lib/api.js";
import Card, { Button, Chip } from "../components/Card.jsx";
import GraphView from "../components/GraphView.jsx";

export default function WorldGraphTab({ embedded }) {
  const { activeId } = useDomain();
  const [data, setData] = useState(null);
  const [filter, setFilter] = useState("");
  const [detail, setDetail] = useState(null);

  const load = useCallback(() => {
    if (activeId) api.graph(activeId, filter || undefined, 250).then(setData);
  }, [activeId, filter]);
  useEffect(load, [load]);

  const onNode = async (nid) => setDetail(await api.node(activeId, nid));
  if (!activeId) return <p className="text-sm text-slate-500">Select or create a domain first.</p>;

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <Card title="World + memory graph" className="lg:col-span-2"
        right={
          <div className="flex items-center gap-2">
            <select value={filter} onChange={(e) => setFilter(e.target.value)}
              className="rounded bg-white px-2 py-1 text-xs">
              <option value="">all edges</option>
              {(data?.rel_types || []).map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
            <Button variant="ghost" onClick={load}>Reload</Button>
            {!embedded && <Button variant="ghost"
              onClick={() => window.open(`/?view=graph&domain=${activeId}`, "_blank", "width=1100,height=800")}>
              Open window ↗</Button>}
          </div>
        }>
        <GraphView data={data} height={embedded ? 640 : 460} onNode={onNode} />
        <div className="mt-2 flex gap-3 text-[11px] text-slate-500">
          <span><span className="text-indigo-400">●</span> memory</span>
          <span><span className="text-emerald-400">●</span> object</span>
          <span><span className="text-amber-400">●</span> world context</span>
          <span className="ml-auto">{data?.nodes?.length ?? 0} nodes · {data?.edges?.length ?? 0} edges</span>
        </div>
      </Card>
      <Card title="Node detail">
        {!detail ? <p className="text-xs text-slate-500">Click a node to inspect its class, provenance, world context, and vector neighbors.</p> : (
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2"><Chip color="indigo">{detail.type}</Chip><span className="font-mono text-[11px] text-slate-500">{detail.id}</span></div>
            {detail.class && <Row k="class" v={detail.class} />}
            {detail.provenance && <Row k="provenance" v={detail.provenance} />}
            {detail.world_ctx_id && <Row k="world context" v={detail.world_ctx_id} />}
            {detail.omega != null && <Row k="ω (importance)" v={Number(detail.omega).toFixed(2)} />}
            {detail.vector_neighbors && (
              <div>
                <div className="mt-1 text-[10px] uppercase tracking-wider text-slate-500">vector neighbors (Qdrant)</div>
                {detail.vector_neighbors.map((n) => (
                  <div key={n.id} className="flex justify-between text-[12px]">
                    <span className="text-slate-700">{n.class}</span><span className="font-mono text-slate-500">{n.sim}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
const Row = ({ k, v }) => (
  <div className="flex justify-between"><span className="text-slate-500">{k}</span><span className="text-slate-800">{v}</span></div>
);
