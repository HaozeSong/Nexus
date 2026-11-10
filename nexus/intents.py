"""Write intents: pending writes exposed to the CC layer without their values, the ordering waits that let readers wait
behind older writers instead of generating over a value that is about to change, and the pricing policy that decides between waiting
and proceeding from the expected costs.

An intent identifies a pending write on an object by task and start time; the buffered value is never exposed (it may still change
while the task continues reasoning). Tasks wait only for older tasks (the age ordering of wound-wait), so the wait-for graph is
acyclic. Registering an intent also invalidates younger tasks that have already read the object when the policy favours waiting
(intent registration also checks for younger tasks).

Pricing (choosing whether to wait): a reader blocked by an older intent proceeds when

    p * C_replay < lambda * t_wait

p = fraction of past intents on the object's table that ended in a committed change, C_replay = mean GPU cost of repairs caused by
such changes, t_wait = expected remaining hold time of the blocking intent (observed hold times of the table minus the intent's age,
clamped at zero), lambda = idle inference capacity, expressed as GPU cost per unit time: 0 when requests are queued (the capacity
would be used by other tasks, so the reader waits), the decode capacity the reader could use when the engine is idle. No time
constant is set by hand: every quantity is a running statistic of this run.
"""
from __future__ import annotations

import time
from collections import defaultdict

from .tools import conflicts


class IntentTable:
    def __init__(self):
        self.intents: dict[str, dict[str, float]] = {}        # object -> {task: start time}
        # pricing statistics, per table
        self.hold_times: dict[str, list] = defaultdict(list)    # observed intent hold times (register -> clear)
        self.ended_change: dict[str, list] = defaultdict(list)  # 1 if the intent ended in a committed change, else 0
        self.repair_cost: dict[str, list] = defaultdict(list)   # GPU cost of repairs caused by committed changes under the table
        self._registered: dict[tuple, float] = {}              # (object, task) -> register time

    # -- registration -------------------------------------------------------------------------------------------
    def register(self, task: str, objs, started: float):
        for o in objs:
            self.intents.setdefault(o, {})[task] = started
            self._registered.setdefault((o, task), time.time())

    def clear(self, task: str, committed: bool, only: set | None = None):
        """Remove the task's intents (all, or those overlapping `only`); record their hold time and outcome for pricing."""
        now = time.time()
        for o in list(self.intents):
            if task not in self.intents[o]:
                continue
            if only is not None and not any(conflicts(o, x) for x in only):
                continue
            t0 = self._registered.pop((o, task), None)
            tbl = o.split(":", 1)[0]
            if t0 is not None:
                self.hold_times[tbl].append(now - t0)
            self.ended_change[tbl].append(1 if committed else 0)
            self.intents[o].pop(task, None)
            if not self.intents[o]:
                del self.intents[o]

    def record_repair_cost(self, obj: str, gpu_s: float):
        self.repair_cost[obj.split(":", 1)[0]].append(float(gpu_s))

    # -- queries --------------------------------------------------------------------------------------------------
    def blocking(self, task: str, started: float, need) -> list:
        """Older tasks' intents overlapping the access set `need`: [(object, holder, holder_start, age)]."""
        out = []
        for o, holders in self.intents.items():
            if any(conflicts(o, x) for x in need):
                for a, t in holders.items():
                    if a != task and t < started:
                        reg = self._registered.get((o, a), time.time())
                        out.append((o, a, t, time.time() - reg))
        return out

    def holders_of(self, task: str) -> set:
        return {o for o, h in self.intents.items() if task in h}

    # -- pricing policy ------------------------------------------------------------------------------------------
    def expected_wait(self, obj: str, age: float) -> float:
        tbl = obj.split(":", 1)[0]
        hs = self.hold_times.get(tbl) or []
        mean_hold = sum(hs) / len(hs) if hs else 0.0
        return max(0.0, mean_hold - age)

    def change_probability(self, obj: str) -> float:
        tbl = obj.split(":", 1)[0]
        xs = self.ended_change.get(tbl) or []
        return (sum(xs) / len(xs)) if xs else 1.0         # no history yet: assume the change will land (conservative)

    def mean_repair_cost(self, obj: str) -> float:
        tbl = obj.split(":", 1)[0]
        xs = self.repair_cost.get(tbl) or []
        return (sum(xs) / len(xs)) if xs else 0.0

    def decide(self, blocking: list, policy: str, telemetry: dict | None, decode_capacity: float) -> str:
        """'wait' or 'proceed' for a reader blocked by `blocking` intents.
        policy: 'wait' (always wait), 'proceed' (always proceed), 'priced' (the rule above)."""
        if policy == "wait" or not blocking:
            return "wait"
        if policy == "proceed":
            return "proceed"
        t = telemetry or {}
        queued = (t.get("queue") or 0) > 0 or (t.get("token_usage") or 0.0) >= 0.95
        lam = 0.0 if queued else decode_capacity
        # the reader proceeds only if proceeding is cheaper than waiting for EVERY blocking intent
        for o, _a, _t0, age in blocking:
            p, c_rep, t_wait = self.change_probability(o), self.mean_repair_cost(o), self.expected_wait(o, age)
            if not (p * c_rep < lam * t_wait):
                return "wait"
        return "proceed"
