"""Short-conflict suites: two agents, one conflict per cell, every protocol.

python scripts/run_short.py --engine sglang --base-url http://127.0.0.1:30001/v1 --model MiniMax-M2.7 --tokenizer <path> \
    --cells workbench,tpcc,tau2 --per-template 10 --protocols serial,naive,twopl,occ,mtpo,sbus,atomix,cordon,memtx,nexus \
    --trials 2 --parallel 4

--per-template K keeps the first K instances of every conflict template; K=10 gives the 48 WorkBench, 50 TPC-C (Micro-short) and
50 tau2 (tau2-short) cells of the paper's short-conflict table. Without it every cell of the suites runs (48 + 90 + 200).

Writes results/short_<model>/trials.jsonl (one record per cell x protocol x trial, with transcripts) and summary.md; reruns resume.
"""
import argparse

from _common import add_engine_args, engine_factory

from benchmarks.registry import first_instances, resolve
from nexus.runner import run_matrix

ap = argparse.ArgumentParser()
add_engine_args(ap)
ap.add_argument("--cells", default="workbench,tpcc,tau2")
ap.add_argument("--per-template", type=int, default=0, help="keep the first K instances of every conflict template (0 = all cells)")
ap.add_argument("--protocols", default="serial,naive,twopl,occ,mtpo,sbus,atomix,cordon,memtx,nexus")
ap.add_argument("--trials", type=int, default=2)
ap.add_argument("--parallel", type=int, default=4)
ap.add_argument("--max-steps", type=int, default=30)
ap.add_argument("--out", default=None)
a = ap.parse_args()

out = a.out or f"results/short_{a.model.replace('/', '_')}"
factory = engine_factory(a)
cells = a.cells.split(",")
if a.per_template:
    cells = [c.qualified for c in first_instances(resolve(cells), a.per_template)]
res = run_matrix(cells, a.protocols.split(","), a.trials, factory, out, a.parallel, a.max_steps)
print(open(f"{out}/summary.md").read())
