"""The Nexus transaction manager: snapshots, versioned read sets, forward validation, write intents.

Every task runs against a snapshot of the committed state overlaid with its own buffered writes; tasks are serialized in
commit order (first committer wins). The manager keeps, per task, a versioned read set (nexus.readsets) and a set of write intents
(nexus.intents). Concurrency control is validation, performed continuously:

  * at every commit the committing task's write set is checked against the read sets of every running task (forward validation,
    ); a task whose read set is invalidated is told so, its in-flight generation is cancelled through the engine's abort
    interface, and its buffered writes derived from the stale observation are retracted; it then repairs by anchored replay
    (nexus.replay) rather than restarting;
  * a write whose premise is stale is refused before it executes (rule 4), an irreversible action validates the whole read set;
  * a pending write registers an intent; a read of an object an older task holds an intent on waits, or proceeds, according
    to the pricing policy; an older task's new intent invalidates younger observers (wound-wait age ordering, acyclic);
  * a task commits only if its read set is valid; a stale read set at commit triggers one more repair, bounded by MAX_REPAIRS.

Configuration flags select the ablations (repair mechanism, intent policy, priorities); the protocol configurations in
protocols/nexus/ (protocol.py and ablations.py) set them.
"""
from __future__ import annotations

import time

from .intents import IntentTable
from .readsets import VersionedReadSet, concrete, hits, stale, stale_for
from .tools import conflicts, is_content_arg
from .transaction import AbortAgent, AgentCtx, CallResult, Protocol

INVALIDATION = ("[INVALIDATION] A concurrent transaction committed changes to: {objs}. What you observed about them is out of "
                "date; the current values are shown below for information only: they do NOT count as reads, so before you write any "
                "of these records again you must read it with a tool (a write based on the shown value is refused). Re-check the "
                "affected steps and redo them if their outcome depends on these values.\n{details}")


def _safe_brief(op) -> str:
    """tool(identifying args) of a withdrawn write; content arguments were derived from outdated data and are not shown."""
    args = {}
    for k, v in (op.kwargs or {}).items():
        args[k] = "<re-derive from the refreshed data>" if is_content_arg(k, v) else v
    return f"{op.tool}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


def _brief_call(tool, kwargs) -> str:
    """tool(identifying args) of a refused call, so the model can tell which call of a batch was refused."""
    def ids(x, key=""):
        if isinstance(x, dict):
            return [v for k, v in x.items() for v in ids(v, k)]
        if isinstance(x, list):
            return [v for e in x for v in ids(e, key)]
        return [f"{key}={x!r}"] if key.endswith("_id") or key in ("id", "item_id", "order_id") else []
    parts = []
    for k, v in (kwargs or {}).items():
        if isinstance(v, (list, dict)):
            inner = ids(v, k)
            parts.append(f"{k}=[{', '.join(inner)}]" if inner else f"{k}=<...>")
        elif isinstance(v, str) and len(v) > 40:
            parts.append(f"{k}=<...>")
        else:
            parts.append(f"{k}={v!r}")
    return f"{tool}({', '.join(parts)})"


class NexusTM(Protocol):
    name = "nexus"
    SHADOW_WRITES = True          # writes are buffered until commit
    PREEMPT = True                # forward validation cancels the affected generation inside the engine
    REPAIR = "replay"             # replay | regen | undo | notify | restart   (repair mechanisms)
    DRAFTS = True                 # replay verifies the prior output as a draft; False = regeneration without drafts
    INTENTS = True                # write intents and ordering waits
    INTENT_POLICY = "priced"      # priced | wait | proceed
    PRIORITIES = True             # engine priority for tasks holding intents that others wait on
    MAX_REPAIRS = 3               # failed commit attempts before the task aborts (liveness bound)
    KEY_WINDOW = 8                # content-keyed sampling window m
    VERIFY_CHUNK = 64             # verification chunk b
    telemetry = None

    def __init__(self, store, agents):
        super().__init__(store, agents)
        self.stats.update({"preemptions": 0, "doomed_continue": 0, "invalidations": 0, "repairs_at_commit": 0,
                           "premise_rechecks": 0, "commits": 0, "intent_waits": 0, "intent_wait_s": 0.0, "intent_proceeds": 0})
        self.commit_ts = 0
        self.committed_writes: list[tuple[int, str]] = []
        self.rs: dict[str, VersionedReadSet] = {a: VersionedReadSet() for a in agents}
        self.intents = IntentTable()
        self.gen_started: dict[str, float] = {}
        self.decode_capacity: float = 0.0       # GPU cost per second the engine could give an idle reader (updated from telemetry)
        for i, c in enumerate(self.agents.values()):
            c.started = time.time() + i * 1e-6
        if self.telemetry is None:
            import nexus.runner as _runner
            self.telemetry = _runner._SAMPLER

    # -- snapshot --------------------------------------------------------------------------------------------------
    def launch_sigma(self, index, agent):
        return float("inf")

    def write_sigma(self, ctx):
        return float("inf")       # buffered ops sit after all committed ops in the author's view; fixed at commit

    def read_exclude(self, ctx):
        return None

    def reset_agent(self, ctx):
        self.rs[ctx.name].reset()
        ctx.withdrawn = []
        ctx.refresh_objs = set()
        ctx.repairs = 0
        ctx.started = time.time()
        self.intents.clear(ctx.name, committed=False)
        self.cv.notify_all()

    def _own_writes(self, agent) -> set:
        mine = set()
        for op in self.store.ops_of(agent):
            mine |= set(op.write_set)
        return mine

    def _lookup(self, ctx):
        return lambda table, rid: self.store.record(table, rid, viewer=ctx.name)

    def priority(self, agent) -> int:
        """Engine scheduling priority for the task's next request: tasks holding intents that other tasks wait on go first (
        'Engine priorities'). 0 = default."""
        if not (self.PRIORITIES and self.INTENTS):
            return 0
        held = self.intents.holders_of(agent)
        if not held:
            return 0
        waiting = sum(1 for o in self.agents.values() if o.name != agent and getattr(o, "waiting_on", None) and any(conflicts(w, h) for w in o.waiting_on for h in held))
        return 1 + waiting

    # -- write intents: registration, early invalidation, ordering waits ----------------------------------------------
    def _register_intents(self, ctx, objs):
        if not self.INTENTS or not objs:
            return
        self.intents.register(ctx.name, objs, ctx.started)
        # a younger running task that already observed one of these objects is invalidated now (its replay waits for the intent to
        # clear before re-reading), unless the policy lets readers proceed on uncommitted state
        if self.INTENT_POLICY == "proceed":
            return
        for y in self.others(ctx):
            if y.committed or y.done or getattr(y, "started", 0) <= getattr(ctx, "started", 0):
                continue
            hit = hits(self.rs[y.name].entries, objs)
            if hit:
                self.stats["early_invalidations"] = self.stats.get("early_invalidations", 0) + 1
                self._withdraw(y, ctx, hit, concrete_objs=objs)

    def _withdraw(self, ctx, writer, objs, concrete_objs=None):
        """Invalidate a younger task's observations of `objs` because an older task is about to write them."""
        self.rs[ctx.name].invalidate(objs)
        withdrawn = self._retract_exact(ctx.name, set(concrete_objs if concrete_objs is not None else objs))
        if withdrawn:
            self.stats["stale_writes_withdrawn"] = self.stats.get("stale_writes_withdrawn", 0) + len(withdrawn)
            ctx.withdrawn = getattr(ctx, "withdrawn", []) + withdrawn
            self._after_retract(ctx.name)
        self.notify(ctx, ("[WITHDRAWN] An older transaction is modifying: " + ", ".join(sorted(objs)) +
                          ". Your observation of these records is withdrawn; the affected steps are rolled back and the records will be "
                          "re-read after that transaction completes. Continue from the refreshed data."
                          + ("\nYour own uncommitted changes to these records were withdrawn as well: " + "; ".join(_safe_brief(o)[:160] for o in withdrawn)
                             + ". Redo them after the re-read if still needed." if withdrawn else "")), objs)
        self._cancel_if_in_flight(ctx, writer, objs, early=True)

    def before(self, ctx, spec, kwargs):
        if not self.INTENTS:
            return
        r, w = spec.declared(kwargs, lookup=self._lookup(ctx))
        need = set(r) | set(w)
        if not need:
            return
        t0 = time.time()
        waited = False
        while True:
            blocking = self.intents.blocking(ctx.name, ctx.started, need)
            if not blocking:
                break
            tele = self.telemetry.latest() if self.telemetry else {}
            if self.intents.decide(blocking, self.INTENT_POLICY, tele, self.decode_capacity) != "wait":
                self.stats["intent_proceeds"] += 1
                self.trace.append({"t": time.time(), "event": "intent_proceed", "agent": ctx.name, "objs": sorted(need)[:6]})
                break
            waited = True
            ctx.waiting_on = need
            self.cv.wait(timeout=5.0)        # woken when intents clear (commit / abort / retract); the timeout only re-prices
        ctx.waiting_on = None
        if waited:
            self.stats["intent_waits"] += 1
            self.stats["intent_wait_s"] = round(self.stats["intent_wait_s"] + (time.time() - t0), 3)
            self.trace.append({"t": time.time(), "event": "intent_wait", "agent": ctx.name, "objs": sorted(need)[:6], "waited_s": round(time.time() - t0, 3)})

    # -- read sets --------------------------------------------------------------------------------------------------
    def _observe(self, ctx, objs):
        self.rs[ctx.name].observe(objs, self.commit_ts, self._own_writes(ctx.name),
                                  lambda t, i: self.store.record(t, i) is not None)

    def after(self, ctx, spec, kwargs, result, op):
        rs = self.rs[ctx.name]
        if op is None:
            r, _ = spec.observed(kwargs, result, [])
            self._observe(ctx, r)
            rs.explicit |= set(r)
            rs.internal -= set(r)
        else:
            new = [o for o in set(op.read_set) | set(op.write_set) if o not in rs.entries]
            self._observe(ctx, op.read_set)
            rs.certify_write(op.write_set, self.commit_ts)
            self._register_intents(ctx, op.write_set)
            rs.internal |= set(new) - rs.explicit

    def begin_generation(self, agent, rid, cancel_event, prompt_reads):
        super().begin_generation(agent, rid, cancel_event, prompt_reads)
        self.gen_started[agent] = time.time()

    def decide(self, victim, writer, objs) -> str:
        return "preempt" if self.PREEMPT else "continue"

    # -- retraction of buffered writes ----------------------------------------------------------------------------------
    def retract_seqs(self, agent, seqs):
        mine = super().retract_seqs(agent, seqs)
        mine += self._take_withdrawn(agent, lambda o: o.seq in set(seqs))
        self._after_retract(agent)
        return mine

    def retract_objs(self, agent, objs):
        mine = super().retract_objs(agent, objs)
        tables = {x.split(":", 1)[0] for x in objs if not x.endswith(("*", "+"))}
        if tables:
            ctx = self.agents[agent]
            with self.cv:
                extra = [o for o in self.store.ops_of(agent) if o not in mine and any(w.endswith("+") and w.split(":", 1)[0] in tables for w in o.write_set)]
                for o in extra:
                    o.undone = True
                    ctx.write_set -= o.write_set
                self.stats["undone_ops"] += len(extra)
                self.cv.notify_all()
            mine += extra
        mine += self._take_withdrawn(agent, lambda o: any(conflicts(w, x) for w in o.write_set for x in objs))
        self._after_retract(agent)
        return mine

    def _retract_exact(self, agent, changed: set) -> list:
        """Undo this task's buffered ops that wrote one of the `changed` objects (rule: retract affected buffered writes). The task's
        own inserts into a table one of whose records changed under it are treated as derived from that record."""
        tables = {x.split(":", 1)[0] for x in changed if not x.endswith(("*", "+"))}

        def hit(op):
            if any(w == x or (x.endswith("*") and conflicts(w, x)) for w in op.write_set for x in changed if not x.endswith("+")):
                return True
            return any(w.endswith("+") and w.split(":", 1)[0] in tables for w in op.write_set)
        ctx = self.agents[agent]
        with self.cv:
            mine = [o for o in self.store.ops_of(agent) if hit(o)]
            for o in mine:
                o.undone = True
                ctx.write_set -= o.write_set
            self.stats["undone_ops"] += len(mine)
            self.cv.notify_all()
        if mine:
            self.trace.append({"agent": agent, "event": "withdraw", "ops": [o.brief() for o in mine]})
        return mine

    def _take_withdrawn(self, agent, pred):
        ctx = self.agents[agent]
        stash = getattr(ctx, "withdrawn", [])
        hit = [o for o in stash if pred(o)]
        ctx.withdrawn = [o for o in stash if o not in hit]
        return hit

    def _after_retract(self, agent):
        with self.cv:
            ctx = self.agents[agent]
            # intents of writes that no longer exist are removed (they were never committed)
            for o in list(self.intents.holders_of(agent)):
                if not any(conflicts(o, w) for w in ctx.write_set):
                    self.intents.clear(agent, committed=False, only={o})
            rs = self.rs[agent]
            live_w = self._own_writes(agent)
            rs.unpin(live_w)
            live = set()
            for op in self.store.ops_of(agent):
                live |= set(op.read_set) | set(op.write_set)
            rs.drop_internal(live)
            self.cv.notify_all()

    # -- invalidation delivery -------------------------------------------------------------------------------------------
    def _refresh(self, objs) -> tuple[str, set]:
        """Current committed values of the invalidated records (information only, never an observation)."""
        lines, shown = [], set()
        for o in sorted(objs)[:8]:
            t, i = o.split(":", 1)
            if i in ("*", "+"):
                lines.append(f"- {o}: a record under this table was created or removed; re-run the listing/search you relied on "
                             f"if your next step depends on it")
                shown.add(o)
                continue
            rec = self.store.record(t, i)
            if rec is None:
                lines.append(f"- {o}: deleted")
            elif len(str(rec)) <= 300:
                lines.append(f"- {o}: {rec}")
            else:
                lines.append(f"- {o}: changed (re-read it to see the current content; do not reconstruct it from memory)")
            shown.add(o)
        return ("\n".join(lines) or "- (see the refreshed tool results)"), shown

    def _cancel_if_in_flight(self, ctx, writer, objs, early=False):
        if ctx.gen_rid is not None and ctx.gen_cancel is not None and hits(ctx.gen_reads, set(objs)):
            action = self.decide(ctx, writer, objs)
            self.trace.append({"t": time.time(), "event": "doomed", "victim": ctx.name, "writer": writer.name, "objs": sorted(objs)[:8],
                               "rid": ctx.gen_rid, "gen_age_s": round(time.time() - self.gen_started.get(ctx.name, time.time()), 3),
                               "action": action, "early": early, "telemetry": (self.telemetry.latest() if self.telemetry else None)})
            if action == "preempt":
                ctx.gen_cancel.set()
                self.stats["preemptions"] += 1
            else:
                self.stats["doomed_continue"] += 1

    def _invalidate(self, ctx, writer: AgentCtx, objs: set, concrete_objs: set | None = None):
        """Deliver an invalidation: mark the entries, retract the buffered writes derived from them, notify (naming the concrete
        records to re-read), and cancel the in-flight generation if its prompt contains the stale value."""
        self.stats["invalidations"] += 1
        self.rs[ctx.name].invalidate(objs)
        withdrawn = self._retract_exact(ctx.name, set(concrete_objs if concrete_objs is not None else objs))
        if withdrawn:
            self.stats["stale_writes_withdrawn"] = self.stats.get("stale_writes_withdrawn", 0) + len(withdrawn)
            ctx.withdrawn = getattr(ctx, "withdrawn", []) + withdrawn
            self._after_retract(ctx.name)
        wild = {o for o in objs if o.endswith(("*", "+"))}
        named = (set(objs) - wild) | {w for w in (concrete_objs or set()) if not w.endswith(("*", "+")) and any(conflicts(w, o) for o in wild)}
        named |= {o for o in wild if not any(conflicts(w, o) for w in named)}
        details, shown = self._refresh(named)
        text = INVALIDATION.format(objs=", ".join(sorted(named)), details=details)
        ctx.refresh_objs = getattr(ctx, "refresh_objs", set()) | shown
        if withdrawn:
            text += ("\nYour own uncommitted changes to these records were withdrawn (they were based on the outdated values): "
                     + "; ".join(_safe_brief(o)[:160] for o in withdrawn) + ". Re-read the records and redo those changes if still needed.")
        self.notify(ctx, text, objs)
        self._cancel_if_in_flight(ctx, writer, objs)

    def _deliver(self, ctx):
        ctx.refresh_objs = set()

    def drain(self, agent):
        notes, objs = super().drain(agent)
        if notes:
            self._deliver(self.agents[agent])
        return notes, objs

    # -- calls: premise check for writes --------------------------------------------------------------------------------
    def call(self, agent, tool, kwargs):
        spec = self.store.backend.specs.get(tool)
        if spec is not None and spec.kind == "write":
            ctx = self.agents[agent]
            with self.cv:
                self.before(ctx, spec, kwargs)
                rs = self.rs[ctx.name].entries
                if spec.unrecoverable:
                    st = stale(rs, self.committed_writes)
                else:
                    _, w = spec.declared(kwargs, lookup=self._lookup(ctx))
                    st = stale_for(rs, set(w), self.committed_writes)
                if st:
                    self.stats["premise_rechecks"] += 1
                    self._invalidate(ctx, ctx, st, concrete_objs=concrete(st, self.committed_writes))
                    notes, ctx.pending = ctx.pending, []
                    objs = set().union(*ctx.pending_objs) if ctx.pending_objs else set()
                    ctx.pending_objs = []
                    self._deliver(ctx)
                    return CallResult(f"Action NOT executed: {_brief_call(tool, kwargs)} — its premise changed (see the invalidation). "
                                      "Read the records named in the invalidation again with a tool, then retry this action if still "
                                      "appropriate; actions whose result was not refused stand.",
                                      notes, 0.0, set(), objs)
        res = super().call(agent, tool, kwargs)
        if res.notifications:
            self._deliver(self.agents[agent])
        return res

    # -- commit: validate, publish in commit order, forward-validate the others -----------------------------------------
    def commit(self, agent):
        ctx = self.agents[agent]
        with self.cv:
            if ctx.pending:
                notes, ctx.pending = ctx.pending, []
                if self.last_note_objs is None:
                    self.last_note_objs = {}
                self.last_note_objs[agent] = set().union(*ctx.pending_objs) if ctx.pending_objs else set()
                ctx.pending_objs = []
                self._deliver(ctx)
                return notes
            st = stale(self.rs[agent].entries, self.committed_writes)
            if st:
                self.stats["repairs_at_commit"] += 1
                ctx.repairs = getattr(ctx, "repairs", 0) + 1
                if ctx.repairs > self.MAX_REPAIRS:
                    self.stats["repair_cap_aborts"] = self.stats.get("repair_cap_aborts", 0) + 1
                    raise AbortAgent(f"read set still stale after {self.MAX_REPAIRS} repairs: {sorted(st)[:4]}")
                self._invalidate(ctx, ctx, st, concrete_objs=concrete(st, self.committed_writes))
                notes, ctx.pending = ctx.pending, []
                if self.last_note_objs is None:
                    self.last_note_objs = {}
                self.last_note_objs[agent] = set().union(*ctx.pending_objs) if ctx.pending_objs else set()
                ctx.pending_objs = []
                self._deliver(ctx)
                return notes
            self.commit_ts += 1
            ts = self.commit_ts
            mine = self.store.ops_of(agent)
            for op in mine:
                op.private = False
                op.sigma = float(ts)
            self.store.commit_agent(agent)
            ctx.committed = True
            self.stats["commits"] += 1
            wset = set()
            for op in mine:
                wset |= set(op.write_set)
            self.intents.clear(agent, committed=bool(wset))
            for w in wset:
                self.committed_writes.append((ts, w))
            for o in self.others(ctx):
                if o.committed or o.done:
                    continue
                hit = hits(self.rs[o.name].entries, wset)
                gen_hit = hits(o.gen_reads, wset) if o.gen_reads else set()
                if hit or gen_hit:
                    self._invalidate(o, ctx, hit | set(gen_hit), concrete_objs=wset)
            self.cv.notify_all()
            return []

    def on_abort(self, ctx):
        self.rs[ctx.name].reset()
        ctx.withdrawn = []
        ctx.refresh_objs = set()
        ctx.repairs = 0
        self.intents.clear(ctx.name, committed=False)
        self.cv.notify_all()

    def on_finish(self, ctx):
        self.intents.clear(ctx.name, committed=False)
        self.cv.notify_all()

    # -- memoized re-execution during replay ----------------------------------------------------
    def unchanged_since(self, objs, ts: int) -> bool:
        """No commit after timestamp `ts` touched any of `objs` (the recorded read's versioned dependencies are unchanged)."""
        return not any(t > ts and any(conflicts(w, o) for o in objs) for t, w in self.committed_writes)

    def memo_read(self, agent, tool, kwargs, result) -> CallResult:
        """Reuse a recorded read result: the read is still registered in the read set (and still subject to ordering waits), only
        the tool is not executed again."""
        spec = self.store.backend.specs[tool]
        ctx = self.agents[agent]
        t0 = time.monotonic()
        with self.cv:
            self.before(ctx, spec, kwargs)
            r, _ = spec.observed(kwargs, result, [])
            ctx.read_set |= r
            self.after(ctx, spec, kwargs, result, None)
            notes, ctx.pending = ctx.pending, []
            note_objs = set().union(*ctx.pending_objs) if ctx.pending_objs else set()
            ctx.pending_objs = []
            self.stats["memo_reads"] = self.stats.get("memo_reads", 0) + 1
            self.cv.notify_all()
        if notes:
            self._deliver(ctx)
        self.trace.append({"t": time.time(), "agent": agent, "tool": tool, "kwargs": kwargs, "result": result[:300], "notes": len(notes), "memo": True})
        return CallResult(result, notes, time.monotonic() - t0, set(r), note_objs, None)

    # -- repair accounting (called by the agent loop) ---------------------------------------------------------------------
    def record_repair(self, agent, objs, gpu_s: float):
        """Feed the pricing statistics with the cost of a repair caused by committed changes on `objs`."""
        for o in objs:
            self.intents.record_repair_cost(o, gpu_s)
