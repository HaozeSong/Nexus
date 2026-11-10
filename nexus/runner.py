"""Run (cell × protocol × trial) and collect two-agent metrics."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agent import run_agent
from .cell import Cell
from .store import ObjectStore
from .engine.telemetry import MetricsSampler
from protocols import load_protocol

_SAMPLER: MetricsSampler | None = None


def start_sampler(root_url: str, interval: float = 1.0):
    """Sample the engine's /metrics in the background for GPU-second attribution and KV timelines."""
    global _SAMPLER
    if root_url and _SAMPLER is None:
        _SAMPLER = MetricsSampler(root_url, interval)
        _SAMPLER.start()
    return _SAMPLER


def attribute_gpu(calls: list[dict]) -> float:
    """GPU-seconds attributable to these LLM calls: each call's wall time weighted by 1/running requests."""
    if _SAMPLER is None:
        return sum(c.get("gen_s", 0.0) for c in calls)
    return sum(_SAMPLER.gpu_seconds(c["t_start"], c["t_end"]) for c in calls if c.get("t_start"))

ROLES = ["A", "B"]


def run_trial(cell: Cell, proto_name: str, llms: dict, trial: int, max_steps: int = 30,
              keep_transcript: bool = True) -> dict:
    P = load_protocol(proto_name)
    store = ObjectStore(cell.backend())
    proto = P(store, ROLES)
    tasks = {"A": cell.task_a, "B": cell.task_b}
    runs = {}
    t0 = time.monotonic()
    tid = f"{cell.name}-{proto_name}-{trial}-{int(time.time())%100000}"
    t_wall0 = time.time()
    if getattr(P, "SEQUENTIAL", False):
        for r in ROLES:
            runs[r] = run_agent(proto, r, tasks[r], llms[r], max_steps, system_extra=cell.system_extra, trial_id=tid)
    else:
        ths = [threading.Thread(target=lambda r=r: runs.__setitem__(r, run_agent(proto, r, tasks[r], llms[r], max_steps, system_extra=cell.system_extra, trial_id=tid)),
                                name=f"{cell.name}-{proto_name}-{trial}-{r}") for r in ROLES]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    wall = time.monotonic() - t0
    inv_ok, detail = cell.invariant(store)
    agents_ok = all(runs[r].status == "ok" for r in ROLES)
    tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "llm_calls": 0}
    for r in ROLES:
        u = runs[r].usage
        tokens["input"] += u.input_tokens; tokens["output"] += u.output_tokens
        tokens["cache_read"] += u.cache_read; tokens["cache_write"] += u.cache_write; tokens["llm_calls"] += u.calls
    tokens["total"] = tokens["input"] + tokens["output"] + tokens["cache_read"] + tokens["cache_write"]
    tokens["wasted_output"] = sum(runs[r].wasted.output_tokens for r in ROLES)
    tokens["wasted_total"] = sum(runs[r].wasted.total for r in ROLES)
    tokens["output_total"] = tokens["output"]
    rec = {
        "cell": cell.qualified, "protocol": proto_name, "trial": trial,
        "correct": bool(inv_ok and agents_ok), "invariant_ok": bool(inv_ok), "detail": detail,
        "agent_status": {r: runs[r].status for r in ROLES},
        "wall_s": round(wall, 3), "tokens": tokens,
        "tool_calls": sum(runs[r].tool_calls for r in ROLES),
        "aborts": sum(runs[r].aborts for r in ROLES),
        "notifications_delivered": sum(runs[r].notifications for r in ROLES),
        "gpu_s": round(attribute_gpu([c for r in ROLES for c in runs[r].calls]), 3),
        "gpu_s_wasted": round(attribute_gpu([c for r in ROLES for c in runs[r].calls if c.get("preempted")]), 3),
        "telemetry": ([{k: v for k, v in smp.items()} for smp in _SAMPLER.samples if t_wall0 - 2 <= smp["t"] <= time.time() + 1] if _SAMPLER else []),
        "preemptions": sum(runs[r].preemptions for r in ROLES),
        "preempted_output_tokens": sum(runs[r].preempted_output_tokens for r in ROLES),
        "rollback_dropped_msgs": sum(runs[r].rollback_dropped_msgs for r in ROLES),
        "calls": [dict(c, agent=r) for r in ROLES for c in runs[r].calls],
        "blocked_s": round(sum(runs[r].blocked_s for r in ROLES), 3),
        "stats": proto.stats,
        "per_agent": {r: {"status": runs[r].status, "wall_s": round(runs[r].wall_s, 3), "tool_calls": runs[r].tool_calls,
                          "aborts": runs[r].aborts, "arg_errors": runs[r].arg_errors, "attempts": runs[r].attempts, "usage": runs[r].usage.asdict(),
                          "final": runs[r].final_text[:300], "error": runs[r].error[:500]} for r in ROLES},
        "ops": [{"seq": o.seq, "sigma": o.sigma, "agent": o.agent, "op": o.brief(), "result": o.result[:120],
                 "undone": o.undone} for o in store.ops],
    }
    if keep_transcript:
        rec["transcript"] = {r: runs[r].transcript for r in ROLES}
        rec["protocol_trace"] = proto.trace
    return rec


def run_matrix(cells: list[str], protocols: list[str], trials: int, llm_factory, out_dir: str,
               parallel: int = 1, max_steps: int = 30, keep_transcript: bool = True, resume: bool = True) -> list[dict]:
    from benchmarks.registry import resolve      # benchmarks build on nexus, so the registry is imported on use
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "trials.jsonl"
    lock = threading.Lock()
    results = []
    done = set()
    if resume and jsonl.exists():
        for line in jsonl.open():
            try:
                r = json.loads(line)
                if r["tokens"].get("llm_calls", 0) == 0:
                    continue          # never reached the LLM (server down): redo it
                results.append(r)
                done.add((r["cell"], r["protocol"], r["trial"]))
            except json.JSONDecodeError:
                pass
    jobs = [(c, p, t) for c in resolve(cells) for p in protocols for t in range(trials) if (c.qualified, p, t) not in done]
    if done:
        print(f"resume: {len(done)} trials already done, {len(jobs)} to go", flush=True)

    def one(job):
        cell, p, t = job
        llms = {r: llm_factory(r) for r in ROLES}
        rec = run_trial(cell, p, llms, t, max_steps, keep_transcript)
        if rec["tokens"]["llm_calls"] == 0 and all("Connection" in v["error"] or "connect" in v["error"].lower()
                                                   for v in rec["per_agent"].values() if v["status"] == "error"):
            raise RuntimeError(f"LLM server unreachable during {cell.qualified}/{p}#{t}: {rec['per_agent']['A']['error'][:120]}")
        with lock:
            results.append(rec)
            with jsonl.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            print(f"[{len(results)}/{len(jobs)}] {cell.name:24s} {p:9s} #{t} correct={rec['correct']} "
                  f"wall={rec['wall_s']:.1f}s tok={rec['tokens']['total']} aborts={rec['aborts']} "
                  f"dl={rec['stats']['deadlocks']} notif={rec['notifications_delivered']} | {rec['detail'][:60]}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        list(ex.map(one, jobs))
    (out / "summary.md").write_text(summarize(results))
    return results


def summarize(results: list[dict]) -> str:
    import statistics as st
    by = {}
    for r in results:
        by.setdefault((r["cell"], r["protocol"]), []).append(r)
    protocols = sorted({r["protocol"] for r in results}, key=lambda p: ["serial", "naive", "2pl", "twopl", "occ", "mtpo", "sbus_ori"].index(p) if p in ["serial", "naive", "2pl", "twopl", "occ", "mtpo", "sbus_ori"] else 99)
    cells = sorted({r["cell"] for r in results})
    lines = ["| cell | protocol | n | correct | wall mean s | wall/serial | tokens mean | tokens/serial | wasted output tok % | aborts/trial | deadlocks/trial | notif/trial | leaked unrec. |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = {p: {"corr": [], "speed": [], "tok": [], "ab": [], "dl": [], "nt": []} for p in protocols}
    for c in cells:
        base = by.get((c, "serial"))
        bw = st.mean(r["wall_s"] for r in base) if base else None
        bt = st.mean(r["tokens"]["total"] for r in base) if base else None
        for p in protocols:
            rs = by.get((c, p))
            if not rs:
                continue
            corr = sum(r["correct"] for r in rs) / len(rs)
            w = st.mean(r["wall_s"] for r in rs); tk = st.mean(r["tokens"]["total"] for r in rs)
            ab = st.mean(r["aborts"] for r in rs); dl = st.mean(r["stats"]["deadlocks"] for r in rs)
            nt = st.mean(r["notifications_delivered"] for r in rs); lk = sum(r["stats"]["leaked_unrecoverable"] for r in rs)
            sp = (bw / w) if bw and w else None; tr = (tk / bt) if bt else None
            wo = sum(r["tokens"].get("wasted_output", 0) for r in rs); oo = sum(r["tokens"]["output"] for r in rs)
            lines.append(f"| {c} | {p} | {len(rs)} | {corr:.0%} | {w:.1f} | {sp and f'{sp:.2f}x' or '-'} | {tk:.0f} | "
                         f"{tr and f'{tr:.2f}x' or '-'} | {(wo / oo if oo else 0):.0%} | {ab:.2f} | {dl:.2f} | {nt:.2f} | {lk} |")
            a = agg[p]; a["corr"].append(corr); a["ab"].append(ab); a["dl"].append(dl); a["nt"].append(nt)
            if sp: a["speed"].append(sp)
            if tr: a["tok"].append(tr)
    lines += ["", "**Mean over cells (normalised to serial per cell)**", "",
              "| protocol | correctness | speedup | token cost | wasted output tok % | aborts/trial | deadlocks/trial | notifications/trial |", "|---|---|---|---|---|---|---|---|"]
    for p in protocols:
        a = agg[p]
        if not a["corr"]:
            continue
        rs_p = [r for r in results if r["protocol"] == p]
        wo = sum(r["tokens"].get("wasted_output", 0) for r in rs_p); oo = sum(r["tokens"]["output"] for r in rs_p)
        lines.append(f"| {p} | {st.mean(a['corr']):.0%} | {st.mean(a['speed']) if a['speed'] else float('nan'):.2f}x | "
                     f"{st.mean(a['tok']) if a['tok'] else float('nan'):.2f}x | {(wo / oo if oo else 0):.0%} | {st.mean(a['ab']):.2f} | {st.mean(a['dl']):.2f} | {st.mean(a['nt']):.2f} |")
    return "\n".join(lines)
