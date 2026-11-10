"""Protocol interface: a toolcall middleware that every agent's calls pass through."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .store import ObjectStore, Op
from .tools import conflicts, ToolSpec


class AbortAgent(Exception):
    """Raised inside a tool call: the agent must clear its context and restart."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class AgentFailed(Exception):
    """Retry cap exceeded (counted as a correctness failure)."""


@dataclass
class CallResult:
    result: str
    notifications: list[str] = field(default_factory=list)
    blocked_s: float = 0.0
    read_objs: set = field(default_factory=set)      # observed read footprint of this call (savepoint metadata)
    note_objs: set = field(default_factory=set)      # objects the attached notifications are about
    op_seq: int | None = None                        # store op sequence number of a write (None for reads)


@dataclass
class AgentCtx:
    name: str
    sigma: float
    read_set: set[str] = field(default_factory=set)
    write_set: set[str] = field(default_factory=set)
    pending: list[str] = field(default_factory=list)   # undelivered notifications
    pending_objs: list[set] = field(default_factory=list)  # objects each pending notification is about
    committed: bool = False
    gen_rid: str | None = None          # in-flight LLM request (Nexus)
    gen_cancel: object = None           # threading.Event set to preempt it
    gen_reads: set = field(default_factory=set)   # objects whose values are inside the in-flight prompt
    aborted_times: int = 0
    done: bool = False


class Protocol:
    """Base: no coordination (== naive).  Subclasses override the hooks."""

    name = "base"
    RETRY_CAP = 5
    SHADOW_WRITES = False   # True: writes stay private to their author until commit publishes them
    RETRACT_TOOL = False    # expose nexus.retract_my_actions (undo own last n writes via their inverses)

    RETRACT_SCHEMA = ("nexus.retract_my_actions",
                      "Undo your own most recent write action(s): the framework reverses their effects on the "
                      "shared systems. Use it when a notification shows that an action you already took no longer "
                      "makes sense (e.g. it modified a record another agent has since cancelled or deleted).",
                      {"type": "object", "properties": {"count": {"type": "integer", "description": "how many of your latest write actions to undo (default 1)"}}})

    def __init__(self, store: ObjectStore, agents: list[str]):
        self.store = store
        self.cv = threading.Condition(store.lock)
        self.agents = {a: AgentCtx(a, sigma=self.launch_sigma(i, a)) for i, a in enumerate(agents)}
        self.stats = {"deadlocks": 0, "aborts": 0, "notifications": 0, "blocked_s": 0.0, "holds": 0,
                      "undone_ops": 0, "leaked_unrecoverable": 0}
        self.trace: list[dict] = []

    # -- policy hooks --------------------------------------------------------------
    def launch_sigma(self, index: int, agent: str) -> float:
        return float("inf")  # everyone sees the live state

    def retract(self, agent: str, count: int = 1) -> CallResult:
        ctx = self.agents[agent]
        with self.cv:
            mine = self.store.ops_of(agent)[-max(1, int(count or 1)):]
            for o in mine:
                o.undone = True
                ctx.write_set -= o.write_set
            self.stats["undone_ops"] += len(mine)
            notes, ctx.pending = ctx.pending, []
            self.cv.notify_all()
        self.trace.append({"agent": agent, "event": "retract", "ops": [o.brief() for o in mine]})
        return CallResult("Retracted: " + ("; ".join(o.brief() for o in mine) or "nothing to retract"), notes, 0.0)

    def retract_seqs(self, agent: str, seqs) -> list:
        """Undo exactly the given (live) ops of this agent — the writes whose tool results a rollback drops."""
        seqs = set(seqs)
        ctx = self.agents[agent]
        with self.cv:
            mine = [o for o in self.store.ops_of(agent) if o.seq in seqs]
            for o in mine:
                o.undone = True
                ctx.write_set -= o.write_set
            self.stats["undone_ops"] += len(mine)
            self.cv.notify_all()
        if mine:
            self.trace.append({"agent": agent, "event": "retract_seqs", "ops": [o.brief() for o in mine]})
        return mine

    def retract_objs(self, agent: str, objs: set) -> list:
        """Selective retract (Nexus surgical rollback): undo only this agent's live ops whose write set touches `objs`."""
        ctx = self.agents[agent]
        with self.cv:
            mine = [o for o in self.store.ops_of(agent) if any(conflicts(w, x) for w in o.write_set for x in objs)]
            for o in mine:
                o.undone = True
                ctx.write_set -= o.write_set
            self.stats["undone_ops"] += len(mine)
            self.cv.notify_all()
        if mine:
            self.trace.append({"agent": agent, "event": "retract_objs", "ops": [o.brief() for o in mine]})
        return mine

    def call(self, agent: str, tool: str, kwargs: dict) -> CallResult:
        if tool == "nexus.retract_my_actions":
            return self.retract(agent, kwargs.get("count", 1))
        spec = self.store.backend.specs[tool]
        ctx = self.agents[agent]
        t0 = time.monotonic()
        with self.cv:
            self.before(ctx, spec, kwargs)          # may block or raise AbortAgent
            if spec.kind == "read":
                result = self.store.read(tool, kwargs, sigma=ctx.sigma, exclude=self.read_exclude(ctx), viewer=agent)
                r, _ = spec.observed(kwargs, result, [])
                ctx.read_set |= r
                op = None
            else:
                op = self.store.write(agent, tool, kwargs, sigma=self.write_sigma(ctx),
                                      exclude=self.read_exclude(ctx), private=self.SHADOW_WRITES)
                result = op.result
                r, w = spec.observed(kwargs, result, op.alloc_ids)
                op.read_set, op.write_set = r, w
                ctx.read_set |= r
                ctx.write_set |= w
            self.after(ctx, spec, kwargs, result, op)  # may raise AbortAgent (eager validation)
            notes, ctx.pending = ctx.pending, []
            note_objs = set().union(*ctx.pending_objs) if ctx.pending_objs else set()
            ctx.pending_objs = []
            self.cv.notify_all()
        blocked = time.monotonic() - t0
        self.trace.append({"t": time.time(), "agent": agent, "tool": tool, "kwargs": kwargs,
                           "result": result[:300], "notes": len(notes), "sigma": ctx.sigma})
        return CallResult(result, notes, blocked, set(r), note_objs, getattr(op, "seq", None) if op is not None else None)

    last_note_objs: dict = None

    def commit(self, agent: str) -> list[str]:
        """Called when the agent produces its final answer.  Returns notifications that
        must be consumed before commit (the agent is re-opened), else [] and commits."""
        ctx = self.agents[agent]
        with self.cv:
            self.validate(ctx)                       # may raise AbortAgent (OCC, S-Bus)
            if ctx.pending:
                notes, ctx.pending = ctx.pending, []
                if self.last_note_objs is None:
                    self.last_note_objs = {}
                self.last_note_objs[agent] = set().union(*ctx.pending_objs) if ctx.pending_objs else set()
                ctx.pending_objs = []
                return notes
            ctx.committed = True
            self.on_commit(ctx)
            self.store.commit_agent(agent)
            self.cv.notify_all()
            return []

    def abort(self, agent: str, reason: str):
        ctx = self.agents[agent]
        with self.cv:
            leaked = sum(1 for o in self.store.ops_of(agent) if self.store.backend.specs[o.tool].unrecoverable)
            n = self.store.undo_agent(agent)
            self.stats["undone_ops"] += n
            self.stats["leaked_unrecoverable"] += leaked
            ctx.read_set.clear(); ctx.write_set.clear(); ctx.pending.clear()
            ctx.aborted_times += 1
            self.stats["aborts"] += 1
            self.on_abort(ctx)
            self.cv.notify_all()
        self.trace.append({"t": time.time(), "agent": agent, "event": "abort", "reason": reason, "undone_ops": n})
        if ctx.aborted_times > self.RETRY_CAP:
            raise AgentFailed(f"{agent} aborted {ctx.aborted_times} times")

    def finish(self, agent: str):
        with self.cv:
            ctx = self.agents[agent]
            ctx.done = True
            if self.SHADOW_WRITES and not ctx.committed:
                # a task that ends without committing (step budget, retry cap, error) leaves no effect: its buffered writes were
                # never published and must not reach the final state
                n = self.store.undo_agent(agent)
                self.stats["abandoned_ops"] = self.stats.get("abandoned_ops", 0) + n
                if n:
                    self.trace.append({"t": time.time(), "agent": agent, "event": "abandoned", "undone_ops": n})
            self.on_finish(ctx)
            self.cv.notify_all()

    # -- overridable ----------------------------------------------------------------
    def read_exclude(self, ctx: AgentCtx) -> set[str] | None:
        return None

    def write_sigma(self, ctx: AgentCtx) -> float | None:
        return None  # physical order

    def before(self, ctx: AgentCtx, spec: ToolSpec, kwargs: dict): ...
    def after(self, ctx: AgentCtx, spec: ToolSpec, kwargs: dict, result: str, op: Op | None): ...
    def validate(self, ctx: AgentCtx): ...
    def on_commit(self, ctx: AgentCtx): ...
    def on_abort(self, ctx: AgentCtx): ...
    def on_finish(self, ctx: AgentCtx): ...

    def others(self, ctx: AgentCtx):
        return [c for c in self.agents.values() if c.name != ctx.name]

    def notify(self, target: AgentCtx, text: str, objs: set | None = None):
        target.pending.append(text)
        target.pending_objs.append(set(objs or ()))
        self.stats["notifications"] += 1

    def requeue(self, agent: str, notes: list, objs: set | None = None):
        """Put back notifications drained by a framework-issued call (a repair's re-read / replay): they must reach the agent
        through its next real tool return, not vanish with the internal CallResult."""
        if not notes:
            return
        ctx = self.agents[agent]
        with self.cv:
            ctx.pending = list(notes) + ctx.pending
            ctx.pending_objs = [set(objs or ())] * len(notes) + ctx.pending_objs

    # -- generation lifecycle (used by Nexus; no-ops otherwise) --------------------------------
    ROLLBACK = "append"    # how a preempted/notified agent resumes: append | truncate | restart

    def begin_generation(self, agent: str, rid: str, cancel_event, prompt_reads: set):
        ctx = self.agents[agent]
        with self.cv:
            ctx.gen_rid, ctx.gen_cancel, ctx.gen_reads = rid, cancel_event, set(prompt_reads)

    def end_generation(self, agent: str):
        ctx = self.agents[agent]
        with self.cv:
            ctx.gen_rid, ctx.gen_cancel, ctx.gen_reads = None, None, set()

    def drain(self, agent: str) -> tuple[list[str], set]:
        """Take pending notifications (after a preemption) and the union of objects they concern."""
        ctx = self.agents[agent]
        with self.cv:
            notes, objs = ctx.pending, set().union(*ctx.pending_objs) if ctx.pending_objs else set()
            ctx.pending, ctx.pending_objs = [], []
            return notes, objs
