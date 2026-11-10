"""Occupancy experiment: run a fixed set of contended cells under controlled background load (batch occupancy)
and record wall, tokens and attributed GPU-seconds per protocol.

Background load = L concurrent long-generation chat requests looping against the same engine (no shared state).
python scripts/occupancy_sweep.py --base-url ... --model ... --levels 0,4,8,16 --protocols serial,occ,mtpo,nexus,nexus_nocancel --trials 2"""
import argparse
import faulthandler
import signal
import os
import json
import threading
import time
from pathlib import Path

from benchmarks.registry import resolve
from nexus.engine.openai_chat import LLM
from nexus.runner import run_trial, start_sampler

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:30001/v1")
ap.add_argument("--model", required=True)
ap.add_argument("--levels", default="0,4,8,16,32")
ap.add_argument("--protocols", default="serial,occ,mtpo,nexus,nexus_nocancel")
ap.add_argument("--cells", default="calendar_lost_update,pm_canary,crm_write_skew,neworder_vs_derived_restock,restock_then_half_order,baggage_lost_update,cancel_vs_bags")
ap.add_argument("--trials", type=int, default=2)
ap.add_argument("--bg-max-tokens", type=int, default=768)
ap.add_argument("--kv-pool", type=int, default=0, help="engine --max-total-tokens in effect (KV-pressure sweep)")
ap.add_argument("--out", default=None)
a = ap.parse_args()
faulthandler.register(signal.SIGUSR1, all_threads=True)   # kill -USR1 <pid> dumps all thread stacks to stderr

root = a.base_url.rstrip("/")[:-3]
sampler = start_sampler(root)
out = Path(a.out or f"results/occ_{a.model}"); out.mkdir(parents=True, exist_ok=True)
jsonl = out / "trials.jsonl"
done = set()
if jsonl.exists():
    for l in jsonl.open():
        try:
            r = json.loads(l); done.add((r["bg_level"], r["cell"], r.get("protocol_id") or {"2pl": "twopl"}.get(r["protocol"], r["protocol"]), r["trial"]))
        except Exception:
            pass

BG_PROMPT = ("Write a detailed, multi-section technical essay (at least 600 words) about the history and internals of "
             "database concurrency control, covering two-phase locking, optimistic concurrency control, multiversion "
             "schemes and snapshot isolation, with examples.")


class Background:
    def __init__(self, level):
        self.level, self.stop, self.threads, self.n_done = level, threading.Event(), [], 0

    def _loop(self):
        llm = LLM(a.model, "openai", a.base_url, "EMPTY", 1.0, a.bg_max_tokens, extra={"top_p": 0.95}, stream=False)
        while not self.stop.is_set():
            try:
                llm.chat("You are a helpful writer.", [{"role": "user", "content": BG_PROMPT}], [])
                self.n_done += 1
            except Exception:
                time.sleep(1)

    def __enter__(self):
        for _ in range(self.level):
            t = threading.Thread(target=self._loop, daemon=True); t.start(); self.threads.append(t)
        time.sleep(20 if self.level else 0)     # let the batch fill before measuring
        return self

    def __exit__(self, *exc):
        self.stop.set()


cells = resolve(a.cells.split(","))
factory = lambda role: LLM(a.model, "openai", a.base_url, "EMPTY", 1.0, 8192, extra={"top_p": 0.95})
for level in [int(x) for x in a.levels.split(",")]:
    with Background(level):
        for c in cells:
            for p in a.protocols.split(","):
                for t in range(a.trials):
                    if (level, c.qualified, p, t) in done:
                        continue
                    try:
                        import urllib.request; urllib.request.urlopen(root + "/health", timeout=10).read()
                    except Exception:
                        print("ERROR engine unreachable; stopping (resume later)", flush=True); os._exit(3)
                    t0 = time.time()
                    rec = run_trial(c, p, {r: factory(r) for r in "AB"}, t, 30, keep_transcript=True)   # the protocol trace is needed for the hazard/doomed events; the transcript is dropped below
                    smp = [s for s in sampler.samples if t0 <= s["t"] <= time.time()] if sampler else []
                    rec.update({"bg_level": level, "running_mean": (sum(s["running"] or 0 for s in smp) / len(smp)) if smp else None,
                                "token_usage_mean": (sum(s["token_usage"] or 0 for s in smp) / len(smp)) if smp else None})
                    rec["kv_pool"] = a.kv_pool
                    rec["protocol_id"] = p
                    rec["c3_events"] = [e for e in (rec.get("protocol_trace") or []) if e.get("event") in ("hazard", "doomed")]
                    rec["telemetry"] = [{k: s.get(k) for k in ("t", "running", "queue", "token_usage")} for s in smp[::2]]
                    rec.pop("transcript", None); rec.pop("protocol_trace", None)
                    with jsonl.open("a") as f:
                        f.write(json.dumps(rec, default=str) + "\n")
                    print(f"L={level:2d} {c.qualified:36s} {p:8s} #{t} correct={rec['correct']} wall={rec['wall_s']:.1f}s "
                          f"tok={rec['tokens']['total']} gpu_s={rec['gpu_s']} wasted_gpu_s={rec['gpu_s_wasted']} running={rec['running_mean']}", flush=True)
print("done", out)
