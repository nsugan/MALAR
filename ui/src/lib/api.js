// REST client for the MALAR multi-domain control plane.
const BASE = import.meta.env.VITE_API_BASE || "/api";

async function req(path, opts = {}) {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  const t = await res.text();
  return t ? JSON.parse(t) : {};
}
const post = (p, body) => req(p, { method: "POST", body: body ? JSON.stringify(body) : undefined });
const put = (p, body) => req(p, { method: "PUT", body: JSON.stringify(body) });
const del = (p) => req(p, { method: "DELETE" });

export const api = {
  // health/legacy
  health: () => req("/health"),
  // domains
  listDomains: () => req("/domains"),
  createDomain: (b) => post("/domains", b),
  selectDomain: (id) => post(`/domains/${id}/select`),
  resetDomain: (id, purge = true) => post(`/domains/${id}/reset?purge_memory=${purge}`),
  deleteDomain: (id, purge = true) => del(`/domains/${id}?purge_memory=${purge}`),
  wipeMemory: () => post(`/domains/wipe-memory`),
  getConfig: (id) => req(`/domains/${id}/config`),
  putConfig: (id, b) => put(`/domains/${id}/config`, b),
  // filesystem browse (pick a server-visible training-data folder)
  browse: (path) => req(`/fs/browse?path=${encodeURIComponent(path || "")}`),
  // train
  analyzeFolder: (id, path) => post(`/domains/${id}/analyze-folder`, { path }),
  selectSubset: (id, report) => post(`/domains/${id}/select-subset`, { report }),
  trainStart: (id) => post(`/domains/${id}/train/start`),
  trainNext: (id) => req(`/domains/${id}/train/next`),
  trainPreview: (id) => req(`/domains/${id}/train/preview`),
  trainConfirm: (id, b) => post(`/domains/${id}/train/confirm`, b),
  trainAuto: (id, n) => post(`/domains/${id}/train/auto`, { n_ticks: n }),
  // knowledge
  learnedValues: (id) => req(`/domains/${id}/learned/values`),
  learnedAffordances: (id) => req(`/domains/${id}/learned/affordances`),
  learnedObjectives: (id) => req(`/domains/${id}/learned/objectives`),
  // graph
  graph: (id, filter, limit = 200) =>
    req(`/domains/${id}/graph?limit=${limit}${filter ? `&filter=${filter}` : ""}`),
  node: (id, nid) => req(`/domains/${id}/node/${nid}`),
  debug: (id) => req(`/domains/${id}/debug`),
  inputs: (id) => req(`/domains/${id}/inputs`),
  inspect: (id, index) => req(`/domains/${id}/inspect/${index}`),
  agents: (id) => req(`/domains/${id}/agents`),
  planDomain: (id, apply) => post(`/domains/${id}/plan`, { apply }),
  setLLMAssist: (id, enabled) => put(`/domains/${id}/llm-assist`, { enabled }),
  getParams: (id) => req(`/domains/${id}/params`),
  setParams: (id, params) => put(`/domains/${id}/params`, { params }),
  // llm observability
  llmStatus: () => req(`/llm/status`),
  llmLog: (limit = 100) => req(`/llm/log?limit=${limit}`),
  llmComplete: (b) => post(`/llm/complete`, b),
  llmRoute: () => req(`/llm/route`),
  llmSetRoute: (b) => put(`/llm/route`, b),
  llmClear: () => post(`/llm/clear`),
  // results
  inferFolder: (id, path) => post(`/domains/${id}/infer/folder`, { path }),

  // Inference & Prediction (V3 probabilistic layer)
  predict: (id, b) => post(`/domains/${id}/predict`, b),
  predictAgents: (id) => req(`/domains/${id}/predict/agents`),
  predictCrossref: (id, b) => post(`/domains/${id}/predict/crossref`, b),
  predictFeedback: (id, b) => post(`/domains/${id}/predict/feedback`, b),
  predictEscalate: (id, b) => post(`/domains/${id}/predict/escalate`, b),

  // Agent Factory (generated agents)
  agentProposals: (id, source) => req(`/domains/${id}/agents/proposals?source=${source}`),
  generateAgent: (b) => post(`/agents/generate`, b),
  listAgents: (domain, cross) => req(`/agents?domain=${encodeURIComponent(domain || "")}&cross=${cross ? 1 : 0}`),
  getAgent: (aid) => req(`/agents/${aid}`),
  validateAgent: (aid, validated) => post(`/agents/${aid}/validate`, { validated }),
  runAgent: (aid, b) => post(`/agents/${aid}/run`, b),
  updateAgentCode: (aid, code, did) => put(`/agents/${aid}/code`, { code, did }),
  testAgent: (aid, b) => post(`/agents/${aid}/test`, b),
  deleteAgent: (aid) => del(`/agents/${aid}`),
  activateAgents: (id, onlyValidated, crossDomain) => post(`/domains/${id}/agents/activate`, { only_validated: onlyValidated, cross_domain: !!crossDomain }),
  agentOutputs: (id) => req(`/domains/${id}/agents/outputs`),
  useAgents: (id, active) => post(`/domains/${id}/agents/use`, { active }),
  agentsStatus: (id) => req(`/domains/${id}/agents/status`),
  results: (id, runId) => req(`/domains/${id}/results/${runId}`),
  compare: (id, runId) => req(`/domains/${id}/results/${runId}/compare`),
};
