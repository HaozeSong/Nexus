"""Shared-workspace tasks: implement one (or two) functions so the module tests pass, record it in CHANGELOG.md (sorted list under
'## Unreleased') and bump the patch version. CHANGELOG/__version__ are read-modify-write hot spots shared by every task of the run;
a module file is shared by every task on that module (12 modules x 4 functions; concurrent pairs are exclusive per function only).
Pair correctness (on the full final state): both tasks' tests pass and both changelog entries are present; the hot spots are judged
at run level (run_metrics)."""
from __future__ import annotations

import re

from nexus.cell import Cell
from benchmarks.workspace.backend import WorkspaceBackend, run_tests
from benchmarks.workspace.repo import BASE_VERSION, FUNCS

CELLS: dict[str, Cell] = {}
PARAMS: dict[str, dict] = {}
SYSTEM_EXTRA = ("You are a software engineer working in a shared repository through file tools (other engineers may be editing it at the "
                "same time). Always read a file right before editing it and use edit_file with an exact old_text; run the module's tests "
                "to verify your implementation. Keep changes minimal and do not touch functions you were not asked to implement.")


def _task(mod, func):
    return (f"Implement `{func}` in pkg/{mod}.py according to its docstring so that tests/test_{mod}.py::test_{func} passes (run the "
            f"tests to verify; leave the other functions of the module as they are). Then record the change in the two release files, "
            f"which you must read first and then rewrite in full with write_file: (a) CHANGELOG.md — add the line "
            f"'- {mod}.{func}: implemented' under the '## Unreleased' heading, keeping the lines under that heading sorted alphabetically "
            f"and everything else unchanged; (b) pkg/__init__.py — bump the patch number of __version__ by one (e.g. 0.4.0 -> 0.4.1).")


def _entry(mod, func):
    return f"- {mod}.{func}: implemented"


def _unreleased(changelog: str) -> list[str]:
    m = re.search(r"## Unreleased\n(.*?)(?:\n## |\Z)", changelog, re.S)
    return [l.strip() for l in (m.group(1) if m else "").splitlines() if l.strip()]


def _bumps(version: str) -> int:
    try:
        return int(version.split(".")[2]) - int(BASE_VERSION.split(".")[2])
    except Exception:
        return -99


def _make_inv(tasks_ab):
    """Pair verdict = the pair's OWN effects survive on the full final state: both functions' tests pass and both changelog lines are
    present. The shared hot spots (version counter, changelog list) are judged at run level by `run_metrics`, because a single slip
    there (by any task, protocol- or model-caused) would otherwise fail every pair of the run."""
    def inv(store):
        st = store.materialize()
        for mod in sorted({m for m, _ in tasks_ab}):
            res = run_tests(st, f"tests/test_{mod}.py")
            for m, f in tasks_ab:
                if m == mod and res.get(f"test_{f}") != "pass":
                    return False, f"test_{f}: {res.get(f'test_{f}', res.get('import_error', res.get('error', 'missing')))[:120]}"
        lines = _unreleased(st.get("CHANGELOG.md", ""))
        missing = [_entry(m, f) for m, f in tasks_ab if _entry(m, f) not in lines]
        if missing:
            return False, f"changelog lost update: missing {missing}"
        return True, f"tests pass, {len(tasks_ab)} changelog entries present"
    return inv


def _ver_of(content: str):
    m = re.search(r'__version__\s*=\s*"([^"]+)"', content or "")
    try:
        return int(m.group(1).split(".")[2]) if m else None
    except Exception:
        return None


def run_metrics(store) -> dict:
    """Run-level integrity of the shared hot spots (lost updates = stale rewrites), counted PER WRITE in commit/replay order so one
    slip counts once: a version write is bad unless it moves the live version FORWARD (a stale rewrite re-derives prev_stale+1, which
    is <= the live version; a chained job may legitimately bump by +L in one write, ); a changelog write is bad if it drops
    a line of the previous live state (a no-op rewrite loses nothing); plus whether the final Unreleased list is sorted."""
    st = store.materialize()
    ops = sorted([o for o in store.live_ops() if o.tool in ("ws.write_file", "ws.edit_file")], key=lambda o: (o.sigma, o.seq))
    base = store.backend.fresh_base()
    prev_v = _ver_of(base["pkg/__init__.py"]); bad_v = 0; n_v = 0
    prev_lines = set(_unreleased(base["CHANGELOG.md"])); bad_c = 0; n_c = 0
    cur = dict(base)
    for o in ops:
        p = o.kwargs.get("path")
        if p not in ("pkg/__init__.py", "CHANGELOG.md"):
            continue
        # apply this op on a running copy to see the file it produced
        store.backend.run_tool(cur, o.tool, dict(o.kwargs), o)
        if p == "pkg/__init__.py":
            n_v += 1
            v = _ver_of(cur.get(p, ""))
            if v is None or prev_v is None or v <= prev_v:
                bad_v += 1
            prev_v = v if v is not None else prev_v
        else:
            n_c += 1
            lines = set(_unreleased(cur.get(p, "")))
            # bad = a previous line dropped (stale rewrite); adding several lines at once (a chained job) or none is not a lost update
            if not prev_lines <= lines:
                bad_c += 1
            prev_lines = lines
    final_lines = _unreleased(st.get("CHANGELOG.md", ""))
    fv = _ver_of(st.get("pkg/__init__.py", ""))
    return {"version_writes": n_v, "version_bad_writes": bad_v, "version_bumps": (fv - int(BASE_VERSION.split(".")[2])) if fv is not None else None,
            "changelog_writes": n_c, "changelog_bad_writes": bad_c, "changelog_lines": len(final_lines), "changelog_sorted": final_lines == sorted(final_lines)}


def _add(name, tasks_a, tasks_b, anomaly):
    def txt(ts):
        if len(ts) == 1:
            return _task(*ts[0])
        return ("Two changes, in order: (1) " + _task(*ts[0]) + " (2) " + _task(*ts[1]))
    cell = Cell(name, anomaly, txt(tasks_a), txt(tasks_b), _make_inv(tasks_a + tasks_b), backend=WorkspaceBackend, benchmark="workspace",
                system_extra=SYSTEM_EXTRA)
    # exclusivity between concurrent pairs is per FUNCTION: module files, CHANGELOG and __version__ are shared by every task of the run
    cell.objects = {f"{m}.{f}" for m, f in tasks_a + tasks_b}
    cell.shared_pool = True      # chained sub-tasks of different agents may touch the same module files
    cell.full_state = True
    cell.run_metrics = run_metrics
    CELLS[name] = cell
    PARAMS[name] = {"a": tasks_a, "b": tasks_b}


mods = list(FUNCS)
M = len(mods)
# same-module pairs: the module file is a shared object as well as CHANGELOG/__init__ (12 modules x 6 pairs = 72 cells)
for mod in mods:
    fs = [f for f, *_ in FUNCS[mod]]
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            _add(f"ws_{mod}_{fs[i]}_{fs[j]}", [(mod, fs[i])], [(mod, fs[j])], "same-module edits + shared changelog/version (RMW hot spots)")
# cross-module pairs (12): only the changelog/version hot spots are shared
for k in range(12):
    m1, m2 = mods[k % M], mods[(k + 1) % M]
    f1 = FUNCS[m1][(k // M) % 4][0]; f2 = FUNCS[m2][(k // M + 2) % 4][0]
    _add(f"ws_x_{m1}_{f1}_{m2}_{f2}", [(m1, f1)], [(m2, f2)], "cross-module edits; shared changelog/version only")
# long tasks (two functions each, 12): longer trajectories with interleaved read-test-edit loops
for k in range(12):
    m1, m2 = mods[k % M], mods[(k + 3) % M]
    fa = [(m1, FUNCS[m1][0][0]), (m2, FUNCS[m2][1][0])]
    fb = [(m1, FUNCS[m1][2][0]), (m2, FUNCS[m2][3][0])]
    _add(f"wsL_{m1}_{m2}_{k}", fa, fb, "two-function tasks on two shared modules + changelog/version")
