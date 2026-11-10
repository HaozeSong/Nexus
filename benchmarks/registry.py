"""Registry of all cells across benchmarks.

A *suite* is one module that exports `CELLS` (name -> Cell). Most benchmark packages provide one suite (`cells.py`);
tau2 and tpcc also provide a long-transaction suite (`long.py`) on the same backend, registered as `tau2long` and
`tpcclong`. A cell is addressed as 'benchmark/cell'; a suite name selects all of its cells.
"""
from __future__ import annotations

import importlib
import sys

from nexus.cell import Cell

SUITES = {
    # short two-agent conflict cells (hand-written invariants)
    "workbench": "benchmarks.workbench.cells",
    "tpcc": "benchmarks.tpcc.cells",
    "tau2": "benchmarks.tau2.cells",
    # long-transaction workloads (serializable-outcome oracle, nexus/serial.py)
    "tau2long": "benchmarks.tau2.long",
    "tpcclong": "benchmarks.tpcc.long",
    "workspace": "benchmarks.workspace.cells",
    "micro": "benchmarks.micro.cells",
}

_cells: dict[str, Cell] | None = None


def load_all() -> dict[str, Cell]:
    """Every cell of every available suite, keyed by 'benchmark/cell' (loaded once)."""
    global _cells
    if _cells is None:
        out = {}
        for suite, modname in SUITES.items():
            try:
                mod = importlib.import_module(modname)
            except (ImportError, OSError) as e:
                # a suite whose vendored dependency (third_party/<repo>, see third_party/README.md) or data is absent is skipped
                sys.stderr.write(f"[benchmarks.registry] suite '{suite}' unavailable ({type(e).__name__}: {str(e)[:80]}); skipped\n")
                continue
            for c in mod.CELLS.values():
                c.group = suite
                out[c.qualified] = c
        _cells = out
    return _cells


def first_instances(cells: list[Cell], k: int) -> list[Cell]:
    """The first `k` instances of every conflict template, in registry order. The short suites instantiate each template on
    several disjoint record sets (tpcc: 5 templates x 18, tau2: 5 x 40, workbench: 7 templates with 4-8 instances); k = 10
    gives 48 WorkBench, 50 TPC-C and 50 tau2 cells."""
    seen: dict[tuple, int] = {}
    out = []
    for c in cells:
        key = (c.benchmark, c.template or c.name)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= k:
            out.append(c)
    return out


def resolve(names: list[str], all_cells: dict[str, Cell] | None = None) -> list[Cell]:
    """Cells selected by `names`: 'all', suite names, qualified names ('tpcc/x') or bare cell names."""
    all_cells = all_cells or load_all()
    if names in (["all"], []):
        return list(all_cells.values())
    out = []
    for n in names:
        if n in all_cells:
            out.append(all_cells[n])
        elif n in SUITES:
            out += [c for c in all_cells.values() if c.group == n]
        else:
            hits = [c for c in all_cells.values() if c.name == n]
            if not hits:
                raise KeyError(f"unknown cell {n}")
            out += hits
    return out
