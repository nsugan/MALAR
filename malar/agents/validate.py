"""Static validation for LLM-generated agent code.

Checks that the code (1) parses as Python 3 and (2) implements the required
``GeneratedAgent.run(ctx)`` interface.

**Capabilities are OPEN (by request):** a generated agent may import and use ANY Python
library and call anything it needs to carry out its task — there is no import allowlist or
call blocklist here, and execution runs with full capabilities (see ``sandbox.py``). This
is powerful and deliberately unsandboxed, so only *validate/activate agents you have
reviewed*. To help that review, the report also returns the list of imports the code uses,
so you can see at a glance what an agent pulls in before approving it.
"""
from __future__ import annotations

import ast


def validate_code(code: str) -> dict:
    """Return {ok, issues, has_interface, imports}.

    `ok` is True when the code parses and defines GeneratedAgent.run — nothing about which
    libraries it uses gates it. `imports` lists top-level modules it imports (for review).
    """
    try:
        tree = ast.parse(code or "")
    except SyntaxError as e:
        return {"ok": False, "issues": [f"syntax error: {e}"], "has_interface": False,
                "imports": []}

    has_interface = False
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.ClassDef) and node.name == "GeneratedAgent":
            for b in node.body:
                if isinstance(b, ast.FunctionDef) and b.name == "run":
                    has_interface = True

    issues = [] if has_interface else ["missing GeneratedAgent.run(ctx) interface"]
    return {"ok": len(issues) == 0, "issues": issues, "has_interface": has_interface,
            "imports": sorted(imports)}
