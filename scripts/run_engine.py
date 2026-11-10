"""Engine effects and overheads.

(a,b) KV-cache pressure: the workspace workload at N=8 against engines started with different KV pools. Each pool is a separate
      server launch (--max-total-tokens); this script runs the sweep against the URLs given, one per pool size:
          python scripts/run_engine.py kv --servers 102k=http://host:30001/v1,64k=http://host:30002/v1,32k=http://host:30003/v1 ...
(c)   Cancelled-output reuse: read off the replay study / concurrency results (draft tokens that came from cancelled generations):
          python scripts/run_engine.py reuse --results results/conc_*_<model>
(d)   CC-layer overhead: wall time per tool call and per commit of the manager alone, measured on the scripted suites (no model):
          python scripts/run_engine.py overhead [--agents 2,4,8,16]
"""
import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

from _common import ROOT, add_engine_args

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["kv", "reuse", "overhead"])
add_engine_args(ap)
ap.add_argument("--servers", default="", help="kv: pool=url list")
ap.add_argument("--benchmark", default="workspace")
ap.add_argument("--protocols", default="serial,twopl,occ,nexus_undo,mtpo,nexus")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--results", default="", help="reuse: glob of result directories")
ap.add_argument("--agents", default="2,4,8,16")
a = ap.parse_args()
PY = sys.executable

if a.what == "kv":
    for item in a.servers.split(","):
        pool, url = item.split("=", 1)
        out = ROOT / "results" / f"kv_{pool}_{a.benchmark}_{a.model.replace('/', '_')}"
        cmd = [PY, str(ROOT / "scripts" / "run_concurrency.py"), "--engine", a.engine, "--base-url", url, "--model", a.model,
               "--benchmark", a.benchmark, "--protocols", a.protocols, "--agents", "8", "--seeds", str(a.seeds), "--out", str(out)]
        if a.tokenizer:
            cmd += ["--tokenizer", a.tokenizer]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)

elif a.what == "reuse":
    tot = {"generated": 0, "reused": 0}
    for d in Path().glob(a.results or "results/conc_*"):
        f = Path(d) / "runs.jsonl"
        if not f.exists():
            continue
        for line in f.open():
            r = json.loads(line)
            if r["protocol_id"] != "nexus":
                continue
            for t in r["tasks"].values():
                for c in t.get("calls") or []:
                    if c.get("preempted"):
                        tot["generated"] += c.get("partial_output", 0)
                    tot["reused"] += c.get("draft_accepted", 0)
    print(json.dumps(tot), f"reuse of cancelled output ~ {tot['reused'] / max(1, tot['generated']):.0%}")

else:
    # CC overhead: time protocol.call / protocol.commit of the manager with N scripted agents on one shared store
    import importlib
    import threading
    from benchmarks.registry import resolve
    from nexus.engine.scripted import ScriptedLLM
    from nexus.runner import run_trial
    from protocols import load_protocol
    P = load_protocol("nexus")
    for n in [int(x) for x in a.agents.split(",")]:
        calls, commits = [], []
        oc, om = P.call, P.commit

        def call(self, ag, tool, kw):
            t0 = time.perf_counter(); r = oc(self, ag, tool, kw); calls.append(time.perf_counter() - t0); return r

        def commit(self, ag):
            t0 = time.perf_counter(); r = om(self, ag); commits.append(time.perf_counter() - t0); return r
        P.call, P.commit = call, commit
        cells = [c for c in resolve(["tpcc"])][:max(2, n)]
        # n concurrent agents: n/2 cells run as concurrent pairs on independent records
        ths = []
        for c in cells[:n // 2 or 1]:
            pol = importlib.import_module("benchmarks.tpcc.scripted").POLICIES[c.name]
            llms = {r: ScriptedLLM(pol(r)) for r in "AB"}
            ths.append(threading.Thread(target=run_trial, args=(c, "nexus", llms, 0, 40)))
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        P.call, P.commit = oc, om
        print(f"N={n:2d}: tool call {1000 * st.mean(calls):.2f} ms (p95 {1000 * sorted(calls)[int(0.95 * len(calls)) - 1]:.2f}), "
              f"commit {1000 * st.mean(commits):.2f} ms  [{len(calls)} calls, {len(commits)} commits; includes the store's own execution]")
