"""Independent (uncontended) WorkBench tasks used as filler load in the mixed workload.

WorkBench ships 690 tasks with ground-truth action lists (data/processed/tasks_and_outcomes/*.csv), e.g.
  calendar.delete_event.func(event_id="00000256")
We execute the ground-truth actions on the base state to obtain the expected row-level effect of the task,
and check that effect in the final state (rows changed/deleted must match; created rows must exist by
content).  Fillers whose ground truth touches records used by the contended cells are excluded.
"""
from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass

from nexus.store import ObjectStore
from benchmarks.workbench.backend import ID_COLS, TOOLS, WORKBENCH, WorkBenchBackend

TASK_DIR = WORKBENCH / "data" / "processed" / "tasks_and_outcomes"
_CALL = re.compile(r"^([a-z_]+\.[a-z_]+)\.func\((.*)\)$", re.S)


def parse_action(s: str):
    m = _CALL.match(s.strip())
    if not m:
        return None
    name, args = m.groups()
    kwargs = {}
    for k, v in re.findall(r'(\w+)=("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,]+)', args):
        try:
            kwargs[k] = ast.literal_eval(v)
        except Exception:
            kwargs[k] = v.strip('"\'')
    return name, kwargs


@dataclass
class Filler:
    id: str
    domain: str
    text: str
    actions: list            # [(tool, kwargs)]
    expected: dict           # table -> {"changed": {id: row}, "deleted": {ids}, "created": [rows]}
    touched: set             # objects "table:id"

    def check(self, store: ObjectStore, ops_filter=None) -> tuple[bool, str]:
        st = store.materialize()
        for table, exp in self.expected.items():
            df = getattr(st, table).fillna("").astype(str)
            col = ID_COLS[table]
            idx = df.set_index(col)
            for rid, row in exp["changed"].items():
                if rid not in idx.index:
                    return False, f"{table}:{rid} missing"
                cur = idx.loc[rid].to_dict()
                if any(str(cur.get(k, "")) != str(v) for k, v in row.items()):
                    return False, f"{table}:{rid} differs"
            for rid in exp["deleted"]:
                if rid in idx.index:
                    return False, f"{table}:{rid} not deleted"
            for row in exp["created"]:
                keys = {k: str(v) for k, v in row.items() if k != col}
                hit = df[(df[list(keys)] == list(keys.values())).all(axis=1)] if keys else df
                if len(hit) == 0:
                    return False, f"{table}: created row missing {list(keys.values())[:2]}"
        return True, "ground-truth effect present"


def _expected_effect(actions):
    store = ObjectStore(WorkBenchBackend())
    base = {t: getattr(store.base, t).fillna("").astype(str).set_index(c) for t, c in ID_COLS.items()}
    for tool, kwargs in actions:
        store.write("gt", tool, kwargs)
    final_state = store.materialize()
    expected, touched = {}, set()
    for t, c in ID_COLS.items():
        f = getattr(final_state, t).fillna("").astype(str).set_index(c)
        changed = {rid: f.loc[rid].to_dict() for rid in f.index if rid in base[t].index and not base[t].loc[rid].equals(f.loc[rid])}
        deleted = {rid for rid in base[t].index if rid not in f.index}
        created = [f.loc[rid].to_dict() for rid in f.index if rid not in base[t].index]
        if changed or deleted or created:
            expected[t] = {"changed": changed, "deleted": deleted, "created": created}
            touched |= {f"{t}:{r}" for r in list(changed) + list(deleted)} | ({f"{t}:*"} if created else set())
    return expected, touched


def load_fillers(domains=("calendar", "email", "project_management", "customer_relationship_manager"),
                 exclude_objects: set[str] | None = None, limit: int | None = None, require_effect=True) -> list[Filler]:
    out = []
    for dom in domains:
        path = TASK_DIR / f"{dom}_tasks_and_outcomes.csv"
        with path.open() as f:
            for i, row in enumerate(csv.DictReader(f)):
                try:
                    acts = [parse_action(a) for a in ast.literal_eval(row["outcome"])]
                except Exception:
                    continue
                acts = [a for a in acts if a and a[0] in TOOLS and not a[0].startswith("analytics")]
                if require_effect and not acts:
                    continue
                if any(a[0].startswith("email.send") or a[0].startswith("email.reply") or a[0].startswith("email.forward") for a in acts):
                    continue   # unrecoverable effects are reserved for the contended cells
                try:
                    expected, touched = _expected_effect(acts)
                except Exception:
                    continue
                if exclude_objects and any(_conf(o, e) for o in touched for e in exclude_objects):
                    continue
                out.append(Filler(f"{dom}#{i}", dom, row["task"], acts, expected, touched))
                if limit and len(out) >= limit:
                    return out
    return out


def _conf(a, b):
    ta, ia = a.split(":", 1)
    tb, ib = b.split(":", 1)
    return ta == tb and (ia == "*" or ib == "*" or (ia == ib and ia != "+"))
