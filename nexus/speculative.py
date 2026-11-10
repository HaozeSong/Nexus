"""Speculative execution substrate shared by the agent CC protocols in this harness.

Execution model (realised on the log-replay ObjectStore):
  * every task gets a launch rank sigma; reads are served from the trajectory prefix with rank <= sigma (filtered read);
  * writes land immediately at their rank (speculative write); a late lower-rank write takes effect at its rank because the
    live copy is always the replay in rank order (undo-suffix/apply/redo without hand-written inverses);
  * a lower-rank write on an object a higher-rank task has read/written produces a dependency notification to that task,
    delivered on its next tool return (with the refreshed value); notifications never flow downwards;
  * unrecoverable tools are held until lower-rank tasks commit; a task cannot commit with pending notifications and, by default,
    waits for lower-rank tasks to commit (global quiescence).
The MTPO protocol of CoAgent (arXiv:2606.15376) is exactly this configuration with LATE_WRITE="undo" and the retract tool
(protocols/mtpo). It is kept as the execution model of that baseline; Nexus (nexus/manager.py) uses commit-order snapshots instead."""
from __future__ import annotations

import time

from nexus.transaction import AgentCtx, Protocol
from nexus.tools import ToolSpec, footprint_conflict


class Speculative(Protocol):
    """Speculative multi-version execution with launch-order ranks and dependency notifications (Nexus's substrate).
    MTPO (CoAgent) is one configuration of this class (protocols/mtpo)."""
    name = "speculative"
    RETRACT_TOOL = True   # A_j may keep, rewrite, or retract its pending operations
    LATE_WRITE = "undo"   # MTPO: undo the higher-sigma suffix, apply, notify the undone agents (they decide to retry);
                          # same-field blind writes above sigma_j shadow the late write (Thomas rule) and are not undone.
                          # "rebase": keep the suffix and re-apply it on top by log replay (framework retries for the agent).

    def launch_sigma(self, index, agent):
        return float(index + 1)

    def write_sigma(self, ctx):
        return ctx.sigma

    # -- hold unrecoverable writes until lower-sigma agents commit ---------------------
    # -- multi-task workloads: every task is a fresh transaction, so it gets a fresh launch rank ------------
    def reset_agent(self, ctx: AgentCtx):
        """Called by the workload driver between an agent's tasks. sigma is a *launch* rank, so a new task gets
        the next rank; keeping the agent's original rank would make every later task of a low-ranked agent block the
        commits of all higher-ranked agents forever (GlobalQuiet waits on lower sigma)."""
        self._launch_counter = getattr(self, "_launch_counter", len(self.agents)) + 1
        ctx.sigma = float(self._launch_counter)

    HOLD_MAX_S = 180.0        # liveness safeguard (not part of MTPO as published): an unrecoverable hold / GlobalQuiet wait gives up after this
    COMMIT_WAIT_MAX_S = 120.0

    def before(self, ctx: AgentCtx, spec: ToolSpec, kwargs: dict):
        if not spec.unrecoverable:
            return
        waited = False
        t0 = time.time()
        while any(o.sigma < ctx.sigma and not (o.committed or o.done) for o in self.others(ctx)):
            waited = True
            if time.time() - t0 > self.HOLD_MAX_S:
                self.stats["wait_timeouts"] = self.stats.get("wait_timeouts", 0) + 1
                self.trace.append({"agent": ctx.name, "event": "hold_timeout", "tool": spec.name})
                break
            self.cv.wait(timeout=0.5)
        if waited:
            self.stats["holds"] += 1
            self.trace.append({"agent": ctx.name, "event": "unrecoverable_released", "tool": spec.name})

    # -- notifications: low sigma writer -> high sigma readers/writers ----------------
    def _blind_same_field(self, late, hi) -> bool:
        """Both ops are field-level blind writes of the same field on the same record (Thomas write rule applies)."""
        f1, f2 = late.kwargs.get("field"), hi.kwargs.get("field")
        return bool(f1) and f1 == f2 and late.tool == hi.tool and set(late.write_set) == set(hi.write_set)

    def after(self, ctx, spec, kwargs, result, op):
        if op is None:
            return
        for o in self.others(ctx):
            if o.sigma <= ctx.sigma:
                continue
            hit_r = footprint_conflict(op.write_set, o.read_set)
            hit_w = footprint_conflict(op.write_set, o.write_set)
            undone = []
            if hit_w and self.LATE_WRITE == "undo":
                # late write: undo the higher-sigma suffix on the conflicting objects (descending sigma), except
                # same-field blind writes, which shadow the late write instead (Thomas write rule)
                for hi in sorted(self.store.ops_of(o.name), key=lambda x: -x.seq):
                    if hi.undone or not footprint_conflict(op.write_set, hi.write_set):
                        continue
                    if self._blind_same_field(op, hi):
                        continue
                    hi.undone = True
                    undone.append(hi)
                    o.write_set -= hi.write_set
                if undone:
                    self.stats["undone_ops"] += len(undone)
                    self.trace.append({"agent": o.name, "event": "late_write_undo", "writer": ctx.name, "undone": [u.brief() for u in undone]})
            if hit_r or hit_w:
                self.notify(o, self._describe(ctx, op, hit_r | hit_w, bool(hit_w) and not undone, o, undone), hit_r | hit_w)

    def _describe(self, writer: AgentCtx, op, objs: set[str], rebased: bool, receiver: AgentCtx, undone=()) -> str:
        import json
        parts = [f"A concurrent agent that is ordered BEFORE you executed {op.brief()} -> \"{op.result[:200]}\"."]
        if undone:
            parts.append("Because that agent's write must take effect before yours, the framework has UNDONE your following "
                         "action(s), which had been applied on stale data: " + "; ".join(u.brief() for u in undone) +
                         ". Their effects are gone. Re-issue them (recomputed from the current data) if they are still needed.")
        for obj in sorted(objs):
            table, rid = obj.split(":", 1)
            if rid in ("*", "+"):
                parts.append(f"The {table} table changed (a record was added or removed); re-run any search you relied on.")
            else:
                before = self.store.record(table, rid, sigma=writer.sigma)
                parts.append(f"{table} record {rid} as left by the agents ordered before you: {json.dumps(before, default=str)}"
                             if before else f"{table} record {rid} no longer exists.")
                if rebased:
                    now = self.store.record(table, rid, sigma=receiver.sigma, viewer=receiver.name)
                    parts.append(f"{table} record {rid} including your own re-applied writes: {json.dumps(now, default=str)}")
        if rebased:
            parts.append("Your own earlier write(s) on these objects were re-applied on top of this change. Check each of "
                         "them against the record as left by the earlier agents: (a) if the record no longer meets the "
                         "condition your task required for modifying it (e.g. it no longer matches the filter you were "
                         "given, or was cancelled/deleted), retract your write(s) with nexus.retract_my_actions; (b) if "
                         "your write computed a value from what you read before, recompute it from the earlier agents' "
                         "record and write it again; (c) otherwise keep it.")
        parts.append("Judge whether this invalidates a premise of your plan; repair only what depends on it, and do not "
                     "redo work that is still valid.")
        return "\n".join(parts)

    # -- commit hook (GlobalQuiet) ----------------------------------------------------
    def commit(self, agent: str):
        ctx = self.agents[agent]
        t0 = time.time()
        with self.cv:
            while True:
                if ctx.pending:
                    notes, ctx.pending = ctx.pending, []
                    if self.last_note_objs is None:
                        self.last_note_objs = {}
                    self.last_note_objs[agent] = set().union(*ctx.pending_objs) if ctx.pending_objs else set()
                    ctx.pending_objs = []
                    return notes
                lower_open = [o.name for o in self.others(ctx) if o.sigma < ctx.sigma and not (o.committed or o.done)]
                if lower_open and time.time() - t0 > self.COMMIT_WAIT_MAX_S:
                    self.stats["wait_timeouts"] = self.stats.get("wait_timeouts", 0) + 1
                    self.trace.append({"t": time.time(), "agent": agent, "sigma": ctx.sigma, "event": "commit_wait_timeout",
                                       "lower_open": [(o.name, o.sigma, o.committed, o.done, len(o.pending)) for o in self.others(ctx) if o.name in lower_open]})
                    lower_open = []
                if not lower_open:
                    ctx.committed = True
                    self.store.commit_agent(agent)
                    self.cv.notify_all()
                    return []
                self.stats["holds"] += 1
                self.cv.wait(timeout=0.5)
