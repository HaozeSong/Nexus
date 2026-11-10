"""Effectiveness of anchored replay.

Runs the long-transaction workloads at N=8 (where invalidations are frequent) with the repair-mechanism configurations and the
sampling / chunk-size variants, and summarises per invalidation: outcome (same tokens, same tool calls, tool call changed), the
share of the invalidated suffix decoded again, draft acceptance, and the GPU time per repair as a function of suffix length.

python scripts/run_replay_study.py --engine sglang ... --benchmarks tau2long,workspace,micro --seeds 3
    [--repairs nexus,nexus_nodraft,nexus_undo,nexus_restart] [--windows 1,2,4,8,16,32] [--chunks 8,16,32,64,128,256]
    [--temperatures 0,0.25,0.5,0.75,1.0]

--repairs "" skips the repair comparison, so that the sensitivity sweeps can be run on their own (e.g. on tau2long only).

Every configuration is one run_concurrency sweep at N=8 (results/replay_<benchmark>_<model>/...); the summary reads them all.
Repairs are compared on the same workload and seeds; the "repair each invalidation from the same snapshot" comparison is
obtained by replaying the recorded trajectories (transcripts carry the repair events with draft/accepted/decoded counts).
"""
import argparse
import json
import subprocess
import sys

from _common import ROOT, add_engine_args

ap = argparse.ArgumentParser()
add_engine_args(ap)
ap.add_argument("--benchmarks", default="tau2long,workspace,micro")
ap.add_argument("--repairs", default="nexus,nexus_nodraft,nexus_undo,nexus_restart,nexus_notify")
ap.add_argument("--windows", default="", help="KEY_WINDOW sweep for the full protocol e.g. 1,2,4,8,16,32")
ap.add_argument("--chunks", default="", help="VERIFY_CHUNK sweep e.g. 8,16,32,64,128,256")
ap.add_argument("--temperatures", default="", help="sampling temperature sweep with content vs position keys")
ap.add_argument("--agents", default="8")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--summary-only", action="store_true")
a = ap.parse_args()

PY = sys.executable
base = [PY, str(ROOT / "scripts" / "run_concurrency.py"), "--engine", a.engine, "--base-url", a.base_url, "--model", a.model,
        "--tool-format", a.tool_format, "--agents", a.agents, "--seeds", str(a.seeds), "--max-tokens", str(a.max_tokens)]
if a.tokenizer:
    base += ["--tokenizer", a.tokenizer]
if a.keyed_engine:
    base += ["--keyed-engine"]


def run(bench, protocols, extra, tag):
    out = ROOT / "results" / f"replay_{tag}_{bench}_{a.model.replace('/', '_')}"
    cmd = base + ["--benchmark", bench, "--protocols", protocols, "--out", str(out)] + extra
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)
    return out


outs = []
if not a.summary_only:
    for b in a.benchmarks.split(","):
        if a.repairs:
            outs.append(run(b, a.repairs, [], "repairs"))
        for m in [x for x in a.windows.split(",") if x]:
            outs.append(run(b, "nexus", ["--set", f"KEY_WINDOW={m}"], f"m{m}"))
        for c in [x for x in a.chunks.split(",") if x]:
            outs.append(run(b, "nexus", ["--set", f"VERIFY_CHUNK={c}"], f"b{c}"))
        for t in [x for x in a.temperatures.split(",") if x]:
            outs.append(run(b, "nexus,nexus_poskeys", ["--temperature", t], f"T{t}"))

# ---- summary over every replay_* result directory -----------------------------------------------------------------------
rows = []
for d in sorted((ROOT / "results").glob("replay_*")):
    f = d / "runs.jsonl"
    if not f.exists():
        continue
    for line in f.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        inval = sum(t.get("replays", 0) for t in r["tasks"].values())
        toks = sum(c.get("draft_verified", 0) for t in r["tasks"].values() for c in (t.get("calls") or []))
        acc = sum(c.get("draft_accepted", 0) for t in r["tasks"].values() for c in (t.get("calls") or []))
        dec = sum(c.get("draft_decoded", 0) for t in r["tasks"].values() for c in (t.get("calls") or []))
        outcomes = [o for t in r["tasks"].values() for o in (t.get("replay_outcomes") or [])]
        same_tokens = sum(1 for o in outcomes if o[0] and o[1] == o[0])
        changed_calls = sum(1 for o in outcomes if o[3] > 0)
        rows.append({"dir": d.name, "benchmark": r.get("benchmark"), "protocol": r["protocol_id"], "overrides": r.get("overrides"),
                     "temperature": None, "n": r["n_agents"], "seed": r["params"]["seed"], "success": r["pair_correct"] / max(1, r["pairs"]),
                     "gpu_s": r.get("gpu_s"), "invalidations": r["stats"].get("invalidations", 0), "replays": inval,
                     "draft_verified": toks, "draft_accepted": acc, "draft_decoded": dec,
                     "acceptance": (acc / toks) if toks else None, "same_tokens": same_tokens, "changed_calls": changed_calls, "outcomes": len(outcomes)})
out = ROOT / "results" / "replay_study_summary.jsonl"
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
by = {}
for r in rows:
    by.setdefault((r["dir"], r["protocol"]), []).append(r)
print("| results | protocol | runs | SCR | replays/run | draft acceptance | decoded again | outcomes: same tokens / tool call changed |")
print("|---|---|---|---|---|---|---|---|")
for k, v in sorted(by.items()):
    ver = sum(x["draft_verified"] for x in v); acc = sum(x["draft_accepted"] for x in v); dec = sum(x["draft_decoded"] for x in v)
    print(f"| {k[0]} | {k[1]} | {len(v)} | {sum(x['success'] for x in v) / len(v):.0%} | {sum(x['replays'] for x in v) / len(v):.1f} | "
          f"{(acc / ver) if ver else float('nan'):.0%} | {(dec / (acc + dec)) if (acc + dec) else float('nan'):.0%} | "
          f"{sum(x['same_tokens'] for x in v)} / {sum(x['changed_calls'] for x in v)} of {sum(x['outcomes'] for x in v)} |")
print(f"\nwritten {out}")
