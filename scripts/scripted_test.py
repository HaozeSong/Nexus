"""Deterministic protocol test without an LLM: python scripts/scripted_test.py [protocols] [cells|benchmark]

Each role is driven by an *ideal self-healing* scripted policy from benchmarks/<b>/scripted.py.
Expected: serial, 2pl (deadlocks), occ (aborts), mtpo (notifications), sbus (409s) and nexus (invalidations, replays) are correct;
naive shows the anomaly."""
import importlib
import os
import sys

from benchmarks.registry import resolve
from nexus.engine.scripted import ScriptedLLM
from nexus.runner import run_trial, summarize

if __name__ == "__main__":
    protocols = sys.argv[1].split(",") if len(sys.argv) > 1 else ["serial", "naive", "twopl", "occ", "mtpo", "sbus", "atomix", "cordon", "memtx", "nexus"]
    cells = resolve(sys.argv[2].split(",") if len(sys.argv) > 2 else ["all"])
    results = []
    for c in cells:
        pol = importlib.import_module(f"benchmarks.{c.benchmark}.scripted").POLICIES[c.name]
        for p in protocols:
            llms = {r: ScriptedLLM(pol(r), gen_time=float(os.environ.get(f"GEN_TIME_{r}", os.environ.get("GEN_TIME", "0")))) for r in "AB"}
            rec = run_trial(c, p, llms, 0, max_steps=40)
            results.append(rec)
            st = rec["stats"]
            print(f"{c.qualified:36s} {p:9s} correct={str(rec['correct']):5s} inv={rec['detail'][:55]:55s} "
                  f"aborts={rec['aborts']} dl={st['deadlocks']} notif={rec['notifications_delivered']} "
                  f"preempt={rec.get('preemptions',0)} ptok={rec.get('preempted_output_tokens',0)} dropped={rec.get('rollback_dropped_msgs',0)} "
                  f"leak={st['leaked_unrecoverable']} status={rec['agent_status']} wall={rec['wall_s']:.2f}s")
            if rec["agent_status"] != {"A": "ok", "B": "ok"}:
                for r in "AB":
                    if rec["per_agent"][r]["error"]:
                        print("   ERROR", r, rec["per_agent"][r]["error"][-400:])
    print()
    print(summarize(results))
