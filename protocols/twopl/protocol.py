"""Strict two-phase locking over declared footprints, with deadlock detection and saga rollback.

Following the CoAgent baseline: "2PL reuses the declarative bindings to acquire read and write locks; a deadlock detector
picks a victim and the saga reverse unwinds it, with the victim's context cleared before it
restarts."  Locks are held until commit (strict 2PL).  A range read (search) takes a shared lock on
"table:*", so a later insert/update inside the range by another agent conflicts (predicate locking),
which is the structural deadlock source the CoAgent baseline describes.
"""
from __future__ import annotations

from nexus.transaction import AbortAgent, AgentCtx, Protocol
from nexus.tools import ToolSpec, conflicts


class P(Protocol):
    name = "2pl"

    def __init__(self, store, agents):
        super().__init__(store, agents)
        self.locks: dict[str, dict] = {}          # obj -> {"S": set(agent), "X": agent|None}
        self.waiting: dict[str, tuple[str, str, int]] = {}  # agent -> (obj, mode, ticket): FIFO wait queue
        self._ticket = 0

    # -- hooks ---------------------------------------------------------------
    def before(self, ctx: AgentCtx, spec: ToolSpec, kwargs: dict):
        r, w = spec.declared(kwargs, lookup=lambda table, rid: self.store.record(table, rid))
        for obj in sorted(r - w):
            self._acquire(ctx, obj, "S")
        for obj in sorted(w):
            self._acquire(ctx, obj, "X")

    def on_commit(self, ctx):
        self._release_all(ctx.name)

    def on_abort(self, ctx):
        self._release_all(ctx.name)

    def on_finish(self, ctx):
        self._release_all(ctx.name)

    # -- lock manager --------------------------------------------------------------
    def _blockers(self, agent: str, obj: str, mode: str, ticket: int | None = None) -> set[str]:
        """Agents whose granted locks conflict with (obj, mode)."""
        out = set()
        for o, lk in self.locks.items():
            if not conflicts(o, obj):
                continue
            if lk["X"] and lk["X"] != agent:
                out.add(lk["X"])
            if mode == "X":
                out |= {a for a in lk["S"] if a != agent}
        return out

    def _holders_wait_for(self, agent: str) -> set[str]:
        if agent not in self.waiting:
            return set()
        obj, mode, ticket = self.waiting[agent]
        return self._blockers(agent, obj, mode, ticket)

    def _deadlock(self, requester: str) -> bool:
        # cycle through requester in the wait-for graph
        stack, seen = list(self._holders_wait_for(requester)), set()
        while stack:
            a = stack.pop()
            if a == requester:
                return True
            if a in seen:
                continue
            seen.add(a)
            stack.extend(self._holders_wait_for(a))
        return False

    def _acquire(self, ctx: AgentCtx, obj: str, mode: str):
        waited = False
        self._ticket += 1
        ticket = self._ticket
        while True:
            if self._holds(ctx.name, obj, mode) or not self._blockers(ctx.name, obj, mode, ticket):
                self._grant(ctx.name, obj, mode)
                self.waiting.pop(ctx.name, None)
                if waited:
                    self.trace.append({"agent": ctx.name, "event": "lock_granted_after_wait", "obj": obj, "mode": mode})
                return
            self.waiting[ctx.name] = (obj, mode, ticket)
            if self._deadlock(ctx.name):
                self.waiting.pop(ctx.name, None)
                self.stats["deadlocks"] += 1
                self.trace.append({"agent": ctx.name, "event": "deadlock_victim", "obj": obj, "mode": mode})
                raise AbortAgent(f"deadlock on {obj} ({mode} lock)")
            waited = True
            self.stats["holds"] += 1
            self.cv.wait(timeout=0.5)

    def _holds(self, agent: str, obj: str, mode: str) -> bool:
        lk = self.locks.get(obj)
        return bool(lk) and (lk["X"] == agent or (mode == "S" and agent in lk["S"]))

    def _grant(self, agent: str, obj: str, mode: str):
        lk = self.locks.setdefault(obj, {"S": set(), "X": None})
        if mode == "X":
            lk["X"] = agent
            lk["S"].discard(agent)
        elif lk["X"] != agent:
            lk["S"].add(agent)

    def _release_all(self, agent: str):
        for o in list(self.locks):
            lk = self.locks[o]
            lk["S"].discard(agent)
            if lk["X"] == agent:
                lk["X"] = None
            if not lk["S"] and lk["X"] is None:
                del self.locks[o]
        self.waiting.pop(agent, None)
        # CoAgent: "A's pending write lock is granted at that same instant" -- grant every pending
        # request that became compatible, in ticket order, before the releasing thread continues, so a
        # restarting victim cannot overtake the waiter it unblocked.
        for a, (wobj, wmode, _t) in sorted(self.waiting.items(), key=lambda kv: kv[1][2]):
            if not self._blockers(a, wobj, wmode):
                self._grant(a, wobj, wmode)
        self.cv.notify_all()
