import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";

const COLORS = { memory: "#6366f1", object: "#10b981", world_context: "#f59e0b", default: "#64748b" };

export default function GraphView({ data, height = 360, onNode }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !data) return;
    const nodes = (data.nodes || []).map((n) => ({
      data: { id: n.id, label: n.label || n.type, type: n.type || "default" },
    }));
    const ids = new Set(nodes.map((n) => n.data.id));
    const edges = (data.edges || [])
      .filter((e) => ids.has(e.source) && ids.has(e.target))
      .map((e, i) => ({ data: { id: `e${i}`, source: e.source, target: e.target, rel: e.rel || "" } }));
    const cy = cytoscape({
      container: ref.current,
      elements: [...nodes, ...edges],
      style: [
        { selector: "node", style: {
          "background-color": (el) => COLORS[el.data("type")] || COLORS.default,
          label: "data(label)", color: "#475569", "font-size": 7, width: 16, height: 16,
          "text-valign": "bottom", "text-margin-y": 2 } },
        { selector: "edge", style: { "line-color": "#cbd5e1", width: 1, "curve-style": "haystack" } },
        { selector: "node:selected", style: { "border-width": 2, "border-color": "#1e293b" } },
      ],
      layout: { name: "cose", animate: false, padding: 20 },
    });
    if (onNode) cy.on("tap", "node", (evt) => onNode(evt.target.id()));
    return () => cy.destroy();
  }, [data, onNode]);
  return <div ref={ref} style={{ height }} className="w-full rounded-lg bg-slate-50" />;
}
