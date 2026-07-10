"""Tests for the dynamic Agent Factory (memory, validation, sandbox, reuse)."""
from __future__ import annotations

import tempfile

from malar.agents.agent_memory import AgentMemory, embed_text
from malar.agents.factory import AgentFactory
from malar.agents.sandbox import run_agent_code
from malar.agents.validate import validate_code

_GOOD = ("class GeneratedAgent:\n"
         "    name = 'Mean'\n"
         "    def run(self, ctx):\n"
         "        import numpy as np\n"
         "        f = np.array(ctx.get('features') or [[0.0]], dtype=float)\n"
         "        return {'mean': float(f.mean()), 'n': int(f.shape[0])}\n")


class _MockLLM:
    def algo(self, prompt, **kw):
        return "1. mean of features; 2. return {'mean': m}"

    def code(self, prompt, **kw):
        return "```python\n" + _GOOD + "```"


def test_validate_allows_any_imports():
    # Capabilities are OPEN (by request): any import is allowed; validation only requires
    # the code to parse and implement the GeneratedAgent.run interface.
    code = ("import os\nclass GeneratedAgent:\n"
            "    def run(self, ctx):\n        return {'cwd_len': len(os.getcwd())}\n")
    rep = validate_code(code)
    assert rep["ok"] and rep["has_interface"]
    assert "os" in rep["imports"]
    # still rejects code missing the required interface
    assert not validate_code("x = 1")["ok"]


def test_validate_accepts_good_interface():
    rep = validate_code(_GOOD)
    assert rep["ok"] and rep["has_interface"]


def test_sandbox_runs_with_full_capabilities():
    out = run_agent_code(_GOOD, {"features": [[1, 2], [3, 4]]})
    assert out["ok"] and out["output"]["mean"] == 2.5
    # arbitrary imports now work — the agent can do anything as tasked
    allowed = run_agent_code(
        "class GeneratedAgent:\n    def run(self, ctx):\n        import os\n"
        "        return {'sep': os.sep}",
        {"features": []})
    assert allowed["ok"] and allowed["output"]["sep"]


def test_embedding_similarity():
    a = embed_text("compute the mean signal")
    b = embed_text("compute mean of the signal")
    c = embed_text("classify virus topology diagram")
    assert float(a @ b) > float(a @ c)


def test_factory_generate_validate_run_reuse():
    db = tempfile.mktemp(suffix=".sqlite")
    mem = AgentMemory(db)
    fac = AgentFactory(mem, _MockLLM())
    g = fac.get_or_generate("MeanAgent", "compute the mean signal of a sample")
    assert g["generated"] and not g["validated"] and g["validation"]["ok"]
    aid = g["agent_id"]
    # pending -> blocked
    assert not fac.run(aid, {"features": [[1, 2]]})["ok"]
    # validate -> runs
    fac.validate_agent(aid)
    r = fac.run(aid, {"features": [[1, 2], [3, 4]]})
    assert r["ok"] and r["output"]["mean"] == 2.5
    # reuse: a similar role reuses the validated agent
    g2 = fac.get_or_generate("MeanAgent2", "compute the mean signal of the sample")
    assert g2["reused"] and g2["agent_id"] == aid


def test_records_last_input_and_output():
    import json
    db = tempfile.mktemp(suffix=".sqlite")
    mem = AgentMemory(db)
    fac = AgentFactory(mem, _MockLLM())
    aid = fac.get_or_generate("MeanAgent", "compute the mean signal of a sample")["agent_id"]
    fac.validate_agent(aid)
    fac.run(aid, {"features": [[1, 2], [3, 4]], "label": "x"})
    a = mem.get(aid)                       # detail row carries the last I/O
    assert json.loads(a["last_output"])["output"]["mean"] == 2.5
    assert "features" in json.loads(a["last_input"])
    # list view stays light (no I/O payload)
    assert "last_output" not in mem.all(with_code=False)[0]
