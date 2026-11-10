"""Re-evaluate workspace runs offline from their recorded ops (replayed in (sigma, seq) order on a fresh base) with the current
invariant and run-level hot-spot metrics; rewrites runs.jsonl in place. Usage: python scripts/reeval_workspace.py results/wl_long2_<m>_workspace/runs.jsonl"""
import json
import re
import sys

from benchmarks.registry import resolve
from benchmarks.workspace.cells import run_metrics
from nexus.store import ObjectStore


def parse(brief):
    m = re.match(r"([\w.]+)\((.*)\)$", brief, re.S)
    tool, args = m.group(1), m.group(2)
    kw = eval(f"dict({args})", {"__builtins__": {}, "dict": dict}) if args.strip() else {}
    return tool, kw


cells = {c.name: c for c in resolve(["workspace"])}
f = sys.argv[1]
rows = [json.loads(l) for l in open(f)]
out = []
for r in rows:
    store = ObjectStore(cells[next(iter(cells))].backend())
    for o in sorted(r["ops"], key=lambda o: (o["sigma"], o["seq"])):
        if o["undone"]:
            continue
        tool, kw = parse(o["op"])
        op = store.write(o["agent"], tool, kw, sigma=float(o["sigma"]), exclude=None, private=False)
        op.task = None
    # the replayed ops carry no task ids, so changelog writers are counted per agent (an upper bound: live writes)
    for pid, d in r["pair_detail"].items():
        c = cells.get(d["cell"].split("/", 1)[-1])
        if c is None:
            continue
        ok, detail = c.invariant(store)
        agents_ok = all(r["tasks"].get(f"{pid}{x}", {}).get("status") == "ok" for x in "ab")
        d["correct"], d["detail"] = bool(ok and agents_ok), detail
    r["pair_correct"] = sum(v["correct"] for v in r["pair_detail"].values())
    r["hot_spot"] = run_metrics(store)
    out.append(r)
    print(r.get("protocol_id") or r["protocol"], "N", r["n_agents"], "seed", r["params"]["seed"], "pairs", r["pair_correct"], "/", r["pairs"], r["hot_spot"])
open(f, "w").write("".join(json.dumps(r) + "\n" for r in out))
