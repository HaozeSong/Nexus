"""Shared-workspace backend (SWE / Terminal-Bench style): a small Python repository edited by several agents through file tools.
State = {path: content}. Tools: list_files, read_file, grep, edit_file (search/replace), write_file (whole file), run_tests (executes
the module's tests in a subprocess on a materialised copy). Objects are files (`files:<path>`); list/grep are table-level reads."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

from nexus.backend import Backend
from nexus.tools import ToolSpec, schemas_from
from benchmarks.workspace.repo import pristine_files

PRISTINE = pristine_files()
_DESC = {
    "ws.list_files": ("List all file paths in the repository.", {"type": "object", "properties": {}, "required": []}),
    "ws.read_file": ("Return the full content of a file.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    "ws.grep": ("Search all files for a regular expression; returns matching lines as path:line: text (max 40).",
                {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}),
    "ws.edit_file": ("Replace ONE exact occurrence of old_text in the file with new_text (old_text must occur exactly once). "
                     "Use it for targeted edits; read the file first so old_text matches exactly.",
                     {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}),
    "ws.write_file": ("Overwrite (or create) a file with the given full content.",
                      {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}),
    "ws.run_tests": ("Run the test functions of one test file (e.g. tests/test_strings.py) against the current repository; returns pass/fail per test with the assertion error.",
                     {"type": "object", "properties": {"test_file": {"type": "string"}}, "required": ["test_file"]}),
}


def _f(path):
    return f"files:{path}"


def _module_of_test(test_file: str) -> str:
    m = re.search(r"test_(\w+)\.py$", test_file or "")
    return f"pkg/{m.group(1)}.py" if m else ""


SPECS = {s.name: s for s in [
    ToolSpec("ws.list_files", "read", "files", footprint=lambda k, r, a: ({"files:*"}, set())),
    ToolSpec("ws.read_file", "read", "files", footprint=lambda k, r, a: ({_f(k.get("path", ""))}, set())),
    ToolSpec("ws.grep", "read", "files", footprint=lambda k, r, a: ({"files:*"}, set())),
    ToolSpec("ws.edit_file", "write", "files", footprint=lambda k, r, a: ({_f(k.get("path", ""))}, {_f(k.get("path", ""))})),
    ToolSpec("ws.write_file", "write", "files", footprint=lambda k, r, a: (set(), {_f(k.get("path", ""))})),
    ToolSpec("ws.run_tests", "read", "files", footprint=lambda k, r, a: ({_f(k.get("test_file", "")), _f(_module_of_test(k.get("test_file", ""))), _f("pkg/__init__.py")}, set())),
]}


def run_tests(files: dict, test_file: str, timeout: float = 20.0) -> dict:
    """Materialise the repository in a temp dir and run every test_* function of the test file in a subprocess."""
    if test_file not in files:
        return {"error": f"{test_file} not found"}
    d = tempfile.mkdtemp(prefix="ws_")
    for p, c in files.items():
        fp = os.path.join(d, p)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp, "w").write(c)
    modname = test_file[:-3].replace("/", ".")
    code = (f"import sys, json, traceback; sys.path.insert(0, {d!r}); out = {{}}\n"
            f"try:\n    import importlib; T = importlib.import_module({modname!r})\n"
            f"except Exception as e:\n    print(json.dumps({{'import_error': traceback.format_exc()[-600:]}})); sys.exit(0)\n"
            f"for n in sorted(dir(T)):\n    if n.startswith('test_'):\n        try:\n            getattr(T, n)(); out[n] = 'pass'\n"
            f"        except Exception as e:\n            out[n] = 'FAIL: ' + traceback.format_exc().strip().splitlines()[-1][:200]\n"
            f"print(json.dumps(out))")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {"error": r.stderr[-400:]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout (infinite loop?)"}
    except Exception as e:
        return {"error": str(e)[:300]}


class WorkspaceBackend(Backend):
    name = "workspace"
    id_cols = {"files": "path"}
    specs = SPECS
    read_before_write = {"ws.write_file"}     # a whole-file rewrite must be grounded in a read still present in the agent's context

    def fresh_base(self):
        return dict(PRISTINE)

    def copy(self, state):
        return dict(state)

    def run_tool(self, state, tool, kwargs, op):
        t = tool.split(".", 1)[1]
        if t == "list_files":
            return "\n".join(sorted(state))
        if t == "read_file":
            p = kwargs.get("path", "")
            return state[p] if p in state else f"Error: {p} not found"
        if t == "grep":
            try:
                rx = re.compile(kwargs.get("pattern", ""))
            except re.error as e:
                return f"Error: bad pattern ({e})"
            out = []
            for p in sorted(state):
                for i, line in enumerate(state[p].splitlines(), 1):
                    if rx.search(line):
                        out.append(f"{p}:{i}: {line}")
                        if len(out) >= 40:
                            return "\n".join(out) + "\n(truncated)"
            return "\n".join(out) or "(no matches)"
        if t == "edit_file":
            p, old, new = kwargs.get("path", ""), kwargs.get("old_text", ""), kwargs.get("new_text", "")
            if p not in state:
                return f"Error: {p} not found"
            n = state[p].count(old)
            if n != 1:
                return f"Error: old_text occurs {n} times in {p}; it must occur exactly once (read the file and copy the text exactly)."
            state[p] = state[p].replace(old, new, 1)
            return f"Edited {p}."
        if t == "write_file":
            p = kwargs.get("path", "")
            state[p] = kwargs.get("content", "")
            return f"Wrote {p} ({len(state[p])} chars)."
        if t == "run_tests":
            return json.dumps(run_tests(state, kwargs.get("test_file", "")))
        return f"Error: unknown tool {tool}"

    def initial_next_ids(self, base):
        return {}

    def record(self, state, table, rid):
        return {"path": rid, "content": state[rid][:600]} if rid in state else None

    # -- write effects: the lines a write added to a file (its own new content, never outdated data) --------------------------
    def write_effect(self, before, after, tool, kwargs):
        p = kwargs.get("path")
        if not p or p not in after:
            return None
        old = set(x.strip() for x in (before.get(p) or "").splitlines() if x.strip())
        added = [x.strip() for x in after[p].splitlines() if x.strip() and x.strip() not in old]
        return {"path": p, "added": added} if added else None

    def effect_holds(self, state, effect):
        cur = set(x.strip() for x in (state.get(effect["path"]) or "").splitlines() if x.strip())
        return all(x in cur for x in effect["added"])

    def effect_brief(self, effect):
        return f"{effect['path']}: it had added the line(s) " + "; ".join(repr(x)[:120] for x in effect["added"][:4]) + (" ..." if len(effect["added"]) > 4 else "")

    def tool_schemas(self, style, names=None):
        return schemas_from(_DESC, style, names or list(_DESC))
