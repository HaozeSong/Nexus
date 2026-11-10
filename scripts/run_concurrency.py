"""Concurrency sweep on the long-transaction workloads (also the transaction-length sweep with --long and the ablation study with the
ablation protocols): N agents in a closed loop over a task pool with contended pairs, several seeds.

python scripts/run_concurrency.py --engine sglang --base-url ... --model ... --tokenizer ... \
    --benchmark tau2long --protocols serial,twopl,occ,nexus_undo,mtpo,nexus --agents 2,4,8,16 --seeds 3 --tasks-per-agent 2

--benchmark: tau2long | workspace | micro | tpcclong | workbench
--long L,...: chain L function-disjoint tasks into one transaction (length sweep)
--set KEY=VALUE: override a protocol class attribute (e.g. --set KEY_WINDOW=16 --set VERIFY_CHUNK=32 for the replay sensitivity sweeps)

Each run appends one record to results/<out>/runs.jsonl: per-task latency, tokens, calls (with draft accounting), protocol stats,
pair verdicts, the op log; reruns resume by (protocol, N, p_c, delta, seed, L). Serial is run at N=2 as the cost reference.
"""
import argparse
import faulthandler
import json
import os
import signal
import statistics as st
from pathlib import Path

from _common import add_engine_args, engine_factory, engine_ok, protocol_overrides

from benchmarks.registry import resolve
from nexus.runner import attribute_gpu
from nexus.workload import build_schedule, run_workload
from protocols import load_protocol

ap = argparse.ArgumentParser()
add_engine_args(ap)
ap.add_argument("--benchmark", default="tau2long")
ap.add_argument("--protocols", default="serial,twopl,occ,nexus_undo,mtpo,nexus")
ap.add_argument("--agents", default="2,4,8,16")
ap.add_argument("--pc", default="1.0", help="contention rate: fraction of tasks that belong to a conflicting pair")
ap.add_argument("--delta", default="empirical", help="release of the second task of a pair: t0 | after_read | mid | pre_commit | empirical")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--tasks", type=int, default=16)
ap.add_argument("--tasks-per-agent", type=int, default=2)
ap.add_argument("--long", default="1")
ap.add_argument("--max-steps", type=int, default=30)
ap.add_argument("--set", action="append", default=[], help="protocol attribute override KEY=VALUE (applied to every protocol)")
ap.add_argument("--out", default=None)
a = ap.parse_args()
faulthandler.register(signal.SIGUSR1, all_threads=True)

cells = resolve([a.benchmark])
backend_cls = cells[0].backend
fillers = []
if a.benchmark == "workbench":
    from benchmarks.workbench.fillers import load_fillers
    fillers = load_fillers(limit=80)
out = Path(a.out or f"results/conc_{a.benchmark}_{a.model.replace('/', '_')}")
out.mkdir(parents=True, exist_ok=True)
jsonl = out / "runs.jsonl"
done = set()
if jsonl.exists():
    for line in jsonl.open():
        try:
            r = json.loads(line)
            done.add((r["protocol_id"], r["n_agents"], r["params"]["p_c"], r["params"]["delta"], r["params"]["seed"], int(r["params"].get("long", 1))))
        except Exception:
            pass
factory = engine_factory(a)
overrides = {}
for kv in a.set:
    k, v = kv.split("=", 1)
    try:
        v = json.loads(v)
    except json.JSONDecodeError:
        pass
    overrides[k] = v

for n in [int(x) for x in a.agents.split(",")]:
    for L in [int(x) for x in a.long.split(",")]:
        for pc in [float(x) for x in a.pc.split(",")]:
            for delta in a.delta.split(","):
                for seed in range(a.seeds):
                    for p in a.protocols.split(","):
                        if p == "serial" and n != 2:
                            continue
                        key = (p, n, pc, delta, seed, L)
                        if key in done:
                            continue
                        if a.engine != "sim" and not engine_ok(a.base_url):
                            print("ERROR engine unreachable; stopping (rerun to resume)", flush=True); os._exit(3)
                        n_tasks = max(a.tasks, a.tasks_per_agent * n) if a.tasks_per_agent else a.tasks
                        sched = build_schedule(cells, fillers, n, n_tasks, pc, delta, seed=seed, empirical="workbench", long=L)
                        P = protocol_overrides(load_protocol(p), overrides)
                        try:
                            res = run_workload(sched, P, backend_cls, factory, a.max_steps, keep_transcript=True, seed=seed)
                        except Exception as e:
                            print(f"ERROR {key}: {type(e).__name__}: {e}", flush=True)
                            continue
                        if res.get("stalled") or (res["tokens_total"] == 0 and len(res["tasks"]) > 0):
                            print(f"ERROR run {key} stalled/empty; not recorded, stopping", flush=True); os._exit(3)
                        res["benchmark"] = a.benchmark
                        res["protocol_id"] = p
                        res["overrides"] = overrides
                        res["gpu_s"] = round(attribute_gpu([c for t in res["tasks"].values() for c in (t.get("calls") or [])]), 3)
                        with jsonl.open("a") as f:
                            f.write(json.dumps(res, default=str) + "\n")
                        print(f"{a.benchmark} {p:14s} N={n} L={L} seed={seed} | makespan={res['makespan_s']:.0f}s p95={res['latency_p95']:.0f}s "
                              f"tok={res['tokens_total']} gpu={res['gpu_s']:.0f}s | pairs {res['pair_correct']}/{res['pairs']} | "
                              f"aborts={res['stats']['aborts']} dl={res['stats']['deadlocks']} inval={res['stats'].get('invalidations', 0)} "
                              f"replays={sum(t.get('replays', 0) for t in res['tasks'].values())}", flush=True)


def summarize(path):
    rs = [json.loads(l) for l in open(path)]
    by = {}
    for r in rs:
        by.setdefault((r["protocol_id"], r["n_agents"], int(r["params"].get("long", 1))), []).append(r)
    lines = ["| protocol | N | L | runs | SCR | GPU s/task | tokens/task | mean JCT s | p95 s | invalidations | replays |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(by.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        succ = sum(x["pair_correct"] for x in v) / max(1, sum(x["pairs"] for x in v))
        nt = sum(len(x["tasks"]) for x in v)
        gpu = sum(x.get("gpu_s", 0) for x in v) / max(1, nt)
        tok = sum(x["tokens_total"] for x in v) / max(1, nt)
        lat = [t["latency_s"] for x in v for t in x["tasks"].values() if "latency_s" in t]
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(v)} | {succ:.0%} | {gpu:.1f} | {tok / 1000:.1f}k | {st.mean(lat) if lat else 0:.0f} | "
                     f"{st.mean(x['latency_p95'] or 0 for x in v):.0f} | {st.mean(x['stats'].get('invalidations', 0) for x in v):.1f} | "
                     f"{st.mean(sum(t.get('replays', 0) for t in x['tasks'].values()) for x in v):.1f} |")
    return "\n".join(lines)


if jsonl.exists():
    (out / "summary.md").write_text(summarize(jsonl))
    print(open(out / "summary.md").read())
