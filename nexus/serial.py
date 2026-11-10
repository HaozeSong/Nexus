"""Serializable-outcome invariants for long transactions.

A long agentic transaction is specified by a *reference action list* (the tool calls an ideal agent would make). For a
conflicting pair (A, B) on shared records the only legal final states are the two serial executions A·B and B·A, computed by
replaying the reference lists on a pristine copy of the backend with the backend's own tools. A run is correct iff
`extract(final state)` equals `extract(A·B)` or `extract(B·A)`; `extract` canonicalises the touched records (id-agnostic
for newly created records). This is the same criterion for every protocol and does not depend on the model's wording."""
from __future__ import annotations

from typing import Callable

from .cell import Cell


class _RefOp:
    """Minimal stand-in for a store Op during reference replays: allocates ids like the store would."""

    def __init__(self, backend, base):
        self.backend, self.counters, self.alloc_ids = backend, dict(backend.initial_next_ids(base)), []

    def next_id(self, table):
        v = self.counters.get(table, 1)
        self.counters[table] = v + 1
        sid = self.backend.format_id(v)
        self.alloc_ids.append(sid)
        return sid


def replay(backend, actions, base=None, op=None):
    """Apply [(tool, kwargs), ...] on a fresh (or given) base state with the backend's tools; returns (state, results)."""
    st = base if base is not None else backend.fresh_base()
    op = op or _RefOp(backend, st)
    out = []
    for a in actions:
        # an action is (tool, kwargs) or a procedure f(state, backend) -> (tool, kwargs) | None that reads the CURRENT state
        # (an ideal agent computes "add one bag" from what it just read, not from the pristine value)
        if callable(a):
            a = a(st, backend)
            if a is None:
                out.append("skipped")
                continue
        tool, kwargs = a
        out.append(backend.run_tool(st, tool, dict(kwargs), op))
    return st, out


def serializable_cell(name: str, benchmark: str, backend_cls, task_a: str, task_b: str, ref_a: list, ref_b: list,
                      extract: Callable, objects: set, anomaly: str = "long transaction pair", system_extra: str = "") -> Cell:
    cache: dict = {}

    def expected():
        if "exp" not in cache:
            be = backend_cls()
            exp = []
            for order, acts in (("A·B", ref_a + ref_b), ("B·A", ref_b + ref_a)):
                st, res = replay(be, acts)
                errs = [r for r in res if isinstance(r, str) and r.startswith("Error")]
                exp.append((order, extract(st), errs))
            cache["exp"] = exp
        return cache["exp"]

    def inv(view):
        got = extract(view.materialize())
        for order, e, errs in expected():
            if got == e:
                return True, f"{order} serial outcome" + (f" (ref errors: {len(errs)})" if errs else "")
        return False, f"not serializable: got {_short(got)}; expected {_short(expected()[0][1])} or {_short(expected()[1][1])}"

    c = Cell(name, anomaly, task_a, task_b, inv, backend=backend_cls, benchmark=benchmark, system_extra=system_extra)
    c.objects = set(objects)
    c.ref_a, c.ref_b, c.extract, c.expected = ref_a, ref_b, extract, expected
    return c


def _short(x, n=220):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "…"
