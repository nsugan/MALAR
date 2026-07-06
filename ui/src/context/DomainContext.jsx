import { createContext, useContext, useEffect, useState, useCallback } from "react";
import { api } from "../lib/api.js";

const Ctx = createContext(null);
export const useDomain = () => useContext(Ctx);

export function DomainProvider({ children }) {
  const [domains, setDomains] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [mode, setMode] = useState("supervised");   // supervised | unsupervised
  const [review, setReview] = useState(true);
  const [debug, setDebug] = useState(false);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const r = await api.listDomains();
      setDomains(r.domains || []);
      setActiveId((cur) => cur || r.active || (r.domains?.[0]?.id ?? null));
    } catch (e) { /* backend may be starting */ }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const select = async (id) => { await api.selectDomain(id); setActiveId(id); refresh(); };
  const active = domains.find((d) => d.id === activeId) || null;

  return (
    <Ctx.Provider value={{ domains, activeId, active, mode, setMode, review, setReview,
      loading, setLoading, refresh, select, setActiveId, debug, setDebug }}>
      {children}
    </Ctx.Provider>
  );
}
