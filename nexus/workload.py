"""Mixed closed-loop workload with controlled contention.

  N agents pull tasks from per-agent queues.  The task pool mixes *filler* tasks (independent, checked
  against their ground-truth effect) and *contended pairs* (a Cell: task A + task B + invariant).  A pair's
  second task is released to another agent under a release condition delta:
      "t0"        both tasks start together
      "after_read"  when agent A has completed its first tool call
      "mid"       when A has completed k tool calls (k = mid_calls)
      "pre_commit"  when A has produced its final answer (commit) -- the conflict hits at the very end
  Contention rate p_c = fraction of pair tasks in the pool; seed makes the schedule reproducible.
"""
from __future__ import annotations

import math
import random
import re
import sys
import threading
import time
from dataclasses import dataclass

from .agent import run_agent
from .cell import Cell
from .store import ObjectStore

# Empirical "reads before first write" distributions (k for delta=empirical).
# WorkBench: 24 models x 690 tasks (results/workbench_trajectory_stats.csv); TraceLab: 15,141 writing requests.
EMPIRICAL_K = {
    "workbench": [0] * 12 + [1] * 30 + [2] * 40 + [3] * 12 + [4] * 4 + [5] * 2,
    "tracelab":  [0] * 8 + [1] * 12 + [2] * 14 + [3] * 12 + [4] * 10 + [5] * 8 + [6] * 6 + [8] * 8 + [12] * 8 + [21] * 8 + [40] * 6,
}
# Tool wall-latency by category from TraceLab (p50, p90 seconds) -> log-normal parameters.
TOOL_LATENCY = {"tracelab": {"read": (0.10, 1.6), "write": (0.06, 1.7), "exec": (0.51, 10.0)}}


@dataclass
class Task:
    tid: str
    text: str
    kind: str                  # "pair_a" | "pair_b" | "filler"
    pair: str | None = None    # pair id
    cell: Cell | None = None
    filler: object = None      # object with .check(store)
    release_after: str | None = None   # tid whose progress gates this task
    release_mode: str = "t0"
    release_calls: int = 0
    arrival_s: float = 0.0             # open-loop arrival offset (poisson mode)


@dataclass
class Schedule:
    tasks: list[Task]
    queues: dict[str, list[Task]]     # agent -> ordered tasks
    n_agents: int
    params: dict


def build_schedule(cells: list[Cell], fillers: list, n_agents: int, n_tasks: int, p_c: float, delta: str,
                   seed: int, mid_calls: int = 2, empirical: str = "workbench", arrival: str = "closed",
                   rate_per_min: float = 0.0, long: int = 1) -> Schedule:
    """delta: t0 | after_read | mid | pre_commit | empirical (k ~ EMPIRICAL_K[empirical]).
    arrival: closed (per-agent queues) | poisson (shared queue, tasks arrive at rate_per_min)."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    queues = {f"A{i}": [] for i in range(n_agents)}
    agents = list(queues)
    if long > 1:
        cells = compose_long(cells, long, rng, want=int(round(p_c * n_tasks / 2)))
    # pair cells must be record-disjoint (two cells on the same reservation/record would interfere), templates mixed at random
    want = int(round(p_c * n_tasks / 2))
    pool = list(cells)
    rng.shuffle(pool)
    pair_cells, pair_objs = [], set()
    for c in pool:
        objs = cell_objects(c)
        if objs & pair_objs:
            continue
        pair_cells.append(c); pair_objs |= objs
        if len(pair_cells) >= want:
            break
    n_pairs = len(pair_cells)
    # fillers must not touch each other's records nor the pair cells' records: contention stays controlled by p_c
    fill, used = [], set()
    pair_ids = {o.split(":", 1)[-1] for o in pair_objs}
    cand = list(fillers)
    rng.shuffle(cand)
    for f in cand:
        touched = getattr(f, "touched", set())
        if any(_conf(a, b) for a in touched for b in used) or any(t.split(":", 1)[-1] in pair_ids for t in touched):
            continue
        fill.append(f)
        used |= touched
    n_fill = max(0, n_tasks - 2 * n_pairs)
    # interleave pairs among fillers at random positions
    slots = ["pair"] * n_pairs + ["fill"] * n_fill
    rng.shuffle(slots)
    fi = 0
    pi = 0
    for k, s in enumerate(slots):
        if s == "fill":
            f = fill[fi % len(fill)] if fill else None
            fi += 1
            if f is None:
                continue
            t = Task(f"f{k}", f.text, "filler", filler=f)
            tasks.append(t)
            queues[agents[k % n_agents]].append(t)
        else:
            c = pair_cells[pi]
            pid = f"p{pi}"
            pi += 1
            a = Task(f"{pid}a", c.task_a, "pair_a", pair=pid, cell=c)
            mode, k = delta, mid_calls
            if delta == "empirical":
                mode, k = "mid", rng.choice(EMPIRICAL_K[empirical])
                if k == 0:
                    mode = "t0"
            b = Task(f"{pid}b", c.task_b, "pair_b", pair=pid, cell=c, release_after=a.tid, release_mode=mode, release_calls=k)
            tasks += [a, b]
            ia = k % n_agents
            ib = (ia + 1 + rng.randrange(max(1, n_agents - 1))) % n_agents if n_agents > 1 else ia
            queues[agents[ia]].append(a)
            queues[agents[ib]].append(b)
    params = {"n_tasks": n_tasks, "p_c": p_c, "delta": delta, "seed": seed, "mid_calls": mid_calls, "arrival": arrival,
              "rate_per_min": rate_per_min, "empirical": empirical, "long": long, "n_pairs": n_pairs}
    if arrival == "poisson":
        # open loop: one shared queue, task i becomes available at Poisson arrival time i
        t = 0.0
        for task in tasks:
            t += rng.expovariate(rate_per_min / 60.0) if rate_per_min > 0 else 0.0
            task.arrival_s = t
        queues = {a: [] for a in agents}
    return Schedule(tasks, queues, n_agents, params)


class Progress:
    """Tracks per-task progress by wrapping the protocol's call/commit."""

    def __init__(self, proto):
        self.proto = proto
        self.cv = threading.Condition()
        self.calls: dict[str, int] = {}
        self.committed: set[str] = set()
        self.done: set[str] = set()
        self.current: dict[str, str] = {}    # agent -> tid
        self.last_call = time.monotonic()
        orig_call, orig_commit = proto.call, proto.commit

        def call(agent, tool, kwargs):
            r = orig_call(agent, tool, kwargs)
            self.last_call = time.monotonic()
            with self.cv:
                tid = self.current.get(agent)
                if tid:
                    self.calls[tid] = self.calls.get(tid, 0) + 1
                    for o in proto.store.ops:
                        if o.agent == agent and o.task is None:
                            o.task = tid
                self.cv.notify_all()
            return r

        def commit(agent):
            r = orig_commit(agent)
            if not r:
                with self.cv:
                    tid = self.current.get(agent)
                    if tid:
                        self.committed.add(tid)
                    self.cv.notify_all()
            return r
        proto.call, proto.commit = call, commit

    def wait_release(self, t: Task, timeout: float = 600):
        if not t.release_after or t.release_mode == "t0":
            return 0.0
        t0 = time.monotonic()
        with self.cv:
            while True:
                dep = t.release_after
                ok = (dep in self.done) or (
                    t.release_mode == "after_read" and self.calls.get(dep, 0) >= 1) or (
                    t.release_mode == "mid" and self.calls.get(dep, 0) >= t.release_calls) or (
                    t.release_mode == "pre_commit" and dep in self.committed)
                if ok or time.monotonic() - t0 > timeout:
                    return time.monotonic() - t0
                self.cv.wait(timeout=0.5)


def cell_objects(cell: Cell) -> set[str]:
    """Records a cell's two tasks touch, for disjoint selection. Each benchmark declares them on its cells (`Cell.objects`);
    a cell that declares none is treated as its own object."""
    return set(cell.objects) if cell.objects else {cell.qualified}


def compose_long(cells: list[Cell], L: int, rng: random.Random, want: int | None = None) -> list[Cell]:
    """Long transactions: each agent task chains L sub-tasks (from L record-disjoint cells, templates as diverse as possible),
    so a pair of agents conflicts L times inside one long context. The pair invariant is the conjunction of the L cell
    invariants. This is the 'agentic workload' regime: many tool calls and a long prompt per transaction.
    Composite cells are mutually record-disjoint; when the greedy packing yields fewer than `want` of them (small pools such as
    the shared workspace), a few more shuffles are tried and the largest packing is kept."""
    if L <= 1:
        return list(cells)
    best = None
    for _ in range(1 if want is None else 24):
        pool = list(cells)
        rng.shuffle(pool)
        out = _compose_once(pool, L)
        if best is None or len(out) > len(best):
            best = out
        if want is None or len(best) >= want:
            break
    return best


def _compose_once(pool: list[Cell], L: int) -> list[Cell]:
    out, used_all = [], set()
    remaining = pool
    while True:
        group, used, tpls = [], set(), set()
        rest = []
        for c in remaining:
            tpl = re.sub(r"_\d+$", "", c.name)
            objs = cell_objects(c)
            if len(group) < L and not (objs & used) and not (objs & used_all) and (tpl not in tpls or L > 5):
                group.append(c); used |= objs; tpls.add(tpl)
            else:
                rest.append(c)
        if len(group) < L:
            break
        used_all |= used
        remaining = rest
        subs = list(group)

        def inv(view, subs=subs):
            oks, det = [], []
            for c in subs:
                ok, d = c.invariant(view)
                oks.append(bool(ok)); det.append(f"{c.name}: {'ok' if ok else d}")
            return all(oks), "; ".join(det)
        head = ("You have several separate requests to handle. Complete ALL of them, in order, finishing one before "
                "starting the next; each request stands on its own.\n")
        ta = head + "\n".join(f"{i + 1}. {c.task_a}" for i, c in enumerate(subs))
        tb = head + "\n".join(f"{i + 1}. {c.task_b}" for i, c in enumerate(subs))
        cell = Cell(f"long{L}_" + "+".join(c.name for c in subs), f"composite of {L} cells", ta, tb, inv,
                    backend=subs[0].backend, benchmark=subs[0].benchmark,
                    system_extra="\n".join(dict.fromkeys(c.system_extra for c in subs if c.system_extra)))
        cell.objects = set(used)
        cell.subs = subs
        for attr in ("full_state", "run_metrics", "shared_pool"):
            if hasattr(subs[0], attr):
                setattr(cell, attr, getattr(subs[0], attr))
        out.append(cell)
    return out


STALL_S = 600.0     # run_workload gives up (stalled=True) after this long without any task completion or tool call


def _lognormal(p50, p90, rng):
    mu = math.log(p50)
    sigma = (math.log(p90) - mu) / 1.2816
    return rng.lognormvariate(mu, sigma)


def _tool_category(spec):
    if spec.kind == "read":
        return "read"
    return "exec" if spec.unrecoverable else "write"


def run_workload(schedule: Schedule, proto_cls, backend_cls, llm_factory, max_steps: int = 30,
                 keep_transcript: bool = False, tool_latency: str | None = None, seed: int = 0) -> dict:
    """Closed loop: every agent works through its queue; one Protocol instance mediates all of them.
    Each task is a fresh agent context (fresh transaction); the protocol keeps per-agent state across tasks,
    so `proto.reset_agent` is called between tasks when the protocol defines it."""
    store = ObjectStore(backend_cls())
    agents = list(schedule.queues)
    proto = proto_cls(store, agents)
    for c in proto.agents.values():
        c.done = True            # idle until the driver launches the agent's first task (_reset_agent flips it back)
    prog = Progress(proto)
    if tool_latency:
        lat_model, lat_rng, lat_lock = TOOL_LATENCY[tool_latency], random.Random(seed), threading.Lock()
        inner_call = proto.call

        def call_with_latency(agent, tool, kwargs):
            r = inner_call(agent, tool, kwargs)
            spec = store.backend.specs.get(tool)
            if spec is not None:
                p50, p90 = lat_model[_tool_category(spec)]
                with lat_lock:
                    d = _lognormal(p50, p90, lat_rng)
                time.sleep(min(d, 30.0))
            return r
        proto.call = call_with_latency
    shared = None
    if schedule.params.get("arrival") == "poisson":
        shared = {"tasks": sorted(schedule.tasks, key=lambda t: t.arrival_s), "i": 0, "cv": threading.Condition()}
    results: dict[str, dict] = {}
    lock = threading.Lock()
    t_start = time.monotonic()

    def next_task(agent):
        if shared is None:
            return None
        with shared["cv"]:
            if shared["i"] >= len(shared["tasks"]):
                return None
            t = shared["tasks"][shared["i"]]
            shared["i"] += 1
        delay = t.arrival_s - (time.monotonic() - t_start)
        if delay > 0:
            time.sleep(delay)
        return t

    progress = {"t": time.monotonic()}

    def worker(agent):
        try:
            _worker(agent)
        except Exception as e:  # never lose a worker silently: release everyone waiting on this agent's tasks
            try:
                proto.finish(agent)          # mark the context done so protocol waits (GlobalQuiet, holds, locks) release
            except Exception:
                pass
            with prog.cv:
                for t in schedule.queues.get(agent, []):
                    prog.done.add(t.tid)
                prog.cv.notify_all()
            with lock:
                results.setdefault(f"worker-{agent}", {"tid": f"worker-{agent}", "agent": agent, "kind": "worker_error", "pair": None,
                                                        "status": "error", "error": f"{type(e).__name__}: {e}", "latency_s": 0,
                                                        "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}})

    seq_lock = threading.Lock() if getattr(proto, "SEQUENTIAL", False) else None   # serial baseline: one task at a time

    def _worker(agent):
        llm = llm_factory(agent)
        it = iter(schedule.queues[agent]) if shared is None else iter(lambda: next_task(agent), None)
        for t in it:
            waited = prog.wait_release(t)
            if seq_lock is not None:
                seq_lock.acquire()
            with prog.cv:
                prog.current[agent] = t.tid
            _reset_agent(proto, agent)
            if hasattr(llm, "begin_task"):
                llm.begin_task(t)
            t0 = time.monotonic()
            try:
                run = run_agent(proto, agent, t.text, llm, max_steps, system_extra=(t.cell.system_extra if t.cell else ""))
            finally:
                if seq_lock is not None:
                    seq_lock.release()
            with prog.cv:
                prog.done.add(t.tid)
                prog.cv.notify_all()
            progress["t"] = time.monotonic()
            with lock:
                results[t.tid] = {"tid": t.tid, "agent": agent, "kind": t.kind, "pair": t.pair, "status": run.status,
                                  "start_s": round(t0 - t_start, 3), "end_s": round(time.monotonic() - t_start, 3),
                                  "latency_s": round(time.monotonic() - t0, 3), "waited_s": round(waited, 3),
                                  "tool_calls": run.tool_calls, "aborts": run.aborts, "notifications": run.notifications,
                                  "usage": run.usage.asdict(), "transcript": run.transcript if keep_transcript else None,
                                  "calls": run.calls, "preemptions": run.preemptions, "preempted_output_tokens": run.preempted_output_tokens,
                                  "wasted_output": run.wasted.output_tokens, "error": (run.error or "")[:600], "arg_errors": run.arg_errors,
                                  "nudges": getattr(run, "nudges", 0), "text": t.text[:200], "repair_choices": getattr(run, "repair_choices", []), "replayed_steps": getattr(run, "replayed_steps", 0),
                                  "redo_nudges": getattr(run, "redo_nudges", 0), "redo_missing_final": getattr(run, "redo_missing_final", 0),
                                  "blind_writes": getattr(run, "blind_writes", 0)}

    ths = [threading.Thread(target=worker, args=(a,), name=f"wl-{a}", daemon=True) for a in agents]   # daemon: a stalled run must not block exit
    for th in ths:
        th.start()
    stalled = False
    while any(th.is_alive() for th in ths):
        for th in ths:
            th.join(timeout=5)
        last = max(progress["t"], max(prog.last_call, t_start))
        if time.monotonic() - last > STALL_S:          # no task finished and no tool call for STALL_S: engine gone or a wait cycle
            stalled = True
            import faulthandler
            sys.stderr.write(f"STALLED run ({proto_cls.name}, N={schedule.n_agents}, params={schedule.params}); thread stacks:\n")
            for a, c in proto.agents.items():
                sys.stderr.write(f"  agent {a}: sigma={c.sigma} committed={c.committed} done={c.done} pending={len(c.pending)} current={prog.current.get(a)}\n")
            sys.stderr.write(f"  tasks done={sorted(prog.done)} results={ {k: v.get('status') for k, v in results.items()} }\n")
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True); sys.stderr.flush()
            break
    makespan = time.monotonic() - t_start
    # correctness
    pairs = {t.pair: t.cell for t in schedule.tasks if t.kind == "pair_a"}
    pair_ok = {}
    for pid, cell in pairs.items():
        ok, detail = cell.invariant(store if getattr(cell, "full_state", False) else store.view({f"{pid}a", f"{pid}b"}))
        agents_ok = all(results.get(f"{pid}{x}", {}).get("status") == "ok" for x in "ab")
        pair_ok[pid] = {"correct": bool(ok and agents_ok), "detail": detail, "cell": cell.qualified}
    fill_ok = {}
    for t in schedule.tasks:
        if t.kind == "filler":
            ok, detail = t.filler.check(store)
            fill_ok[t.tid] = {"correct": bool(ok and results.get(t.tid, {}).get("status") == "ok"), "detail": detail}
    hot = None
    rm = next((getattr(t.cell, "run_metrics", None) for t in schedule.tasks if t.cell is not None and getattr(t.cell, "run_metrics", None)), None)
    if rm is not None:
        try:
            hot = rm(store)
        except Exception as e:
            hot = {"error": str(e)[:200]}
    tok = sum(r["usage"]["input"] + r["usage"]["output"] + r["usage"]["cache_read"] + r["usage"]["cache_write"] for r in results.values())
    lat = sorted(r["latency_s"] for r in results.values() if "latency_s" in r)
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None
    return {"protocol": proto_cls.name, "backend": backend_cls.name, "params": schedule.params, "n_agents": schedule.n_agents,
            "makespan_s": round(makespan, 3), "throughput_tasks_per_min": round(60 * len(results) / makespan, 3),
            "latency_p50": q(0.5), "latency_p95": q(0.95), "tokens_total": tok,
            "pair_correct": sum(v["correct"] for v in pair_ok.values()), "pairs": len(pair_ok),
            "filler_correct": sum(v["correct"] for v in fill_ok.values()), "fillers": len(fill_ok),
            "stats": proto.stats, "tasks": results, "pair_detail": pair_ok, "filler_detail": fill_ok, "stalled": stalled, "hot_spot": hot,
            "events": [e for e in proto.trace if e.get("event") in ("commit_wait_timeout", "hold_timeout", "hazard", "doomed", "deadlock")][:500],
            "ops": [{"seq": o.seq, "sigma": o.sigma, "agent": o.agent, "op": o.brief(), "undone": o.undone} for o in store.ops]}


def _conf(a, b):
    ta, ia = a.split(":", 1)
    tb, ib = b.split(":", 1)
    return ta == tb and (ia == "*" or ib == "*" or (ia == ib and ia != "+"))


def _reset_agent(proto, agent):
    """Fresh transaction context for the agent's next task."""
    ctx = proto.agents[agent]
    with proto.cv:
        ctx.read_set.clear(); ctx.write_set.clear(); ctx.pending.clear()
        ctx.committed = False; ctx.done = False; ctx.aborted_times = 0
        if hasattr(proto, "reset_agent"):
            proto.reset_agent(ctx)
        proto.cv.notify_all()
