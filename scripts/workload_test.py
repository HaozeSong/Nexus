"""End-to-end test of the mixed closed-loop workload with scripted agents (no LLM).
python scripts/workload_test.py [protocols] [n_agents] [n_tasks] [p_c] [delta]"""
import importlib
import json
import sys

from benchmarks.registry import resolve
from benchmarks.workbench.backend import WorkBenchBackend
from benchmarks.workbench.fillers import load_fillers
from nexus.engine.scripted import ScriptedLLM
from nexus.workload import build_schedule, run_workload
from protocols import load_protocol


class TaskSwitchingLLM(ScriptedLLM):
    """One scripted LLM per agent; the policy changes with the task (pair role from task.kind, fillers = replay GT)."""

    def __init__(self):
        super().__init__(lambda m: None)

    def begin_task(self, t):
        self.seen_notes = 0
        if t.kind == "filler":
            acts = list(t.filler.actions)

            def pol(messages, acts=acts):
                from nexus.engine.scripted import tool_history
                done = len(tool_history(messages))
                return [acts[done]] if done < len(acts) else None
            self.policy = pol
        else:
            mod = importlib.import_module(f"benchmarks.{t.cell.benchmark}.scripted")
            self.policy = mod.POLICIES[t.cell.name]("A" if t.kind == "pair_a" else "B")


if __name__ == "__main__":
    protocols = sys.argv[1].split(",") if len(sys.argv) > 1 else ["naive", "mtpo", "occ"]
    n_agents = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n_tasks = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    p_c = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    delta = sys.argv[5] if len(sys.argv) > 5 else "t0"
    arrival = sys.argv[6] if len(sys.argv) > 6 else "closed"
    tool_latency = sys.argv[7] if len(sys.argv) > 7 else None
    cells = resolve(["workbench"])
    excl = {"calendar_events:00000256", "calendar_events:00000100", "calendar_events:00000275", "crm_data:00000107",
            "crm_data:00000141", "crm_data:00000087", "crm_data:00000181", "project_tasks:00000149", "project_tasks:00000151",
            "project_tasks:00000300"}
    fillers = load_fillers(exclude_objects=excl, limit=40)
    print(f"{len(fillers)} fillers, {len(cells)} cells")
    for p in protocols:
        sched = build_schedule(cells, fillers, n_agents, n_tasks, p_c, delta, seed=1, arrival=arrival, rate_per_min=120)
        res = run_workload(sched, load_protocol(p), WorkBenchBackend, lambda a: TaskSwitchingLLM(), max_steps=40,
                           tool_latency=(tool_latency or None))
        print(f"{p:9s} agents={n_agents} tasks={n_tasks} p_c={p_c} delta={delta} | makespan={res['makespan_s']:.1f}s "
              f"thr={res['throughput_tasks_per_min']:.1f}/min p50={res['latency_p50']:.2f}s p95={res['latency_p95']:.2f}s | "
              f"pairs {res['pair_correct']}/{res['pairs']} fillers {res['filler_correct']}/{res['fillers']} | "
              f"aborts={res['stats']['aborts']} dl={res['stats']['deadlocks']} notif={res['stats']['notifications']}")
        bad = {k: v for k, v in res["pair_detail"].items() if not v["correct"]}
        badf = {k: v for k, v in res["filler_detail"].items() if not v["correct"]}
        if bad or badf:
            print("   pair failures:", json.dumps(bad)[:300]); print("   filler failures:", json.dumps(badf)[:300])
