"""Log-replay object store (backend-agnostic).

Objects are strings "table:id" (one record) or "table:*" (the whole table / range).  A write op is
recorded as (sigma, seq, agent, tool, kwargs, allocated ids).  The state a reader at rank `sigma` should
observe is materialise(sigma) = base + replay of every op with op.sigma <= sigma, in (sigma, seq) order.
The live state is materialise(+inf).  Aborting an agent = dropping its ops (the saga inverse).  Shadow
(private) ops are visible only to their author until a protocol publishes them.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .backend import Backend


@dataclass
class Op:
    seq: int
    sigma: float
    agent: str
    tool: str
    kwargs: dict
    store: "ObjectStore"
    alloc_ids: list = field(default_factory=list)
    alloc_cursor: int = -1
    replaying: bool = False
    result: str = ""
    read_set: set = field(default_factory=set)
    write_set: set = field(default_factory=set)
    undone: bool = False   # dropped from the trajectory (abort)
    private: bool = False  # shadow write: visible only to its author until commit
    task: str | None = None  # workload task id (set by the workload runner)
    committed: bool = False  # published by a successful commit; never undone by a later abort of the same agent

    def next_id(self, table: str) -> str:
        """Stable id allocation: first execution draws from the store counter, replays reuse it."""
        self.alloc_cursor += 1
        if self.replaying:
            return self.alloc_ids[self.alloc_cursor]
        new_id = self.store._alloc_id(table)
        self.alloc_ids.append(new_id)
        return new_id

    def brief(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items() if v not in (None, ""))
        return f"{self.tool}({args})"


class ObjectStore:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.base = backend.fresh_base()
        self.ops: list[Op] = []
        self.lock = threading.RLock()
        self._seq = 0
        self._next_id = backend.initial_next_ids(self.base)
        self.events: list[dict] = []

    # -- ids ---------------------------------------------------------------
    def _alloc_id(self, table: str) -> str:
        v = self._next_id[table]
        self._next_id[table] = v + 1
        return self.backend.format_id(v)

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # -- replay --------------------------------------------------------------
    def _run(self, op: Op, state, replaying: bool) -> str:
        op.replaying = replaying
        op.alloc_cursor = -1
        try:
            return str(self.backend.run_tool(state, op.tool, op.kwargs, op))
        finally:
            op.replaying = False

    def live_ops(self):
        return [o for o in self.ops if not o.undone]

    def materialize(self, sigma: float = float("inf"), exclude: set[str] | None = None, viewer: str | None = None,
                    before_seq: int | None = None):
        """State at rank `sigma`: base + ops with op.sigma <= sigma in (sigma, seq) order.  `exclude` drops
        every op of those agents; private ops are visible only to their author (`viewer`); viewer=None sees all.
        `before_seq` keeps only ops with seq < before_seq (the view an op was executed on)."""
        with self.lock:
            st = self.backend.copy(self.base)
            ops = [o for o in self.live_ops() if o.sigma <= sigma and (not exclude or o.agent not in exclude)
                   and (not o.private or viewer is None or o.agent == viewer) and (before_seq is None or o.seq < before_seq)]
            for o in sorted(ops, key=lambda o: (o.sigma, o.seq)):
                self._run(o, st, replaying=True)
            return st

    def effect_of(self, op: Op, viewer: str):
        """The durable effect of `op` (typically a write just retracted) in its author's view: the backend compares the object
        before and after the op. None when the backend cannot say (the redo is then tracked by identity only)."""
        try:
            with self.lock:
                before = self.materialize(viewer=viewer, before_seq=op.seq)
                after = self.backend.copy(before)
                self._run(op, after, replaying=True)
                return self.backend.write_effect(before, after, op.tool, op.kwargs)
        except Exception:
            return None

    # -- execution -------------------------------------------------------------
    def read(self, tool: str, kwargs: dict, sigma: float = float("inf"), exclude: set[str] | None = None,
             viewer: str | None = None) -> str:
        with self.lock:
            st = self.materialize(sigma, exclude, viewer)
            op = Op(seq=-1, sigma=sigma, agent=viewer or "?", tool=tool, kwargs=kwargs, store=self)
            return self._run(op, st, replaying=False)

    def write(self, agent: str, tool: str, kwargs: dict, sigma: float | None = None,
              exclude: set[str] | None = None, private: bool = False) -> Op:
        """Execute a write at rank `sigma` (default: physical order) and append it to the log.  The result is
        what the agent observes (op run on top of the prefix <= sigma as seen by that agent); the live state is
        the replay in sigma order, so a lower-sigma write arriving late takes effect before higher ones."""
        with self.lock:
            seq = self.next_seq()
            if sigma is None:
                sigma = float(seq)
            st = self.materialize(sigma, exclude, viewer=agent)
            op = Op(seq=seq, sigma=sigma, agent=agent, tool=tool, kwargs=kwargs, store=self, private=private)
            op.result = self._run(op, st, replaying=False)
            self.ops.append(op)
            return op

    def undo_agent(self, agent: str) -> int:
        """Saga rollback of the agent's *uncommitted* ops (ops of its earlier, committed tasks stay)."""
        with self.lock:
            n = 0
            for o in self.ops:
                if o.agent == agent and not o.undone and not o.committed:
                    o.undone = True
                    n += 1
            return n

    def ops_of(self, agent: str) -> list[Op]:
        """The agent's live, not-yet-committed ops (its current transaction)."""
        return [o for o in self.live_ops() if o.agent == agent and not o.committed]

    def commit_agent(self, agent: str):
        with self.lock:
            for o in self.ops:
                if o.agent == agent and not o.undone:
                    o.committed = True

    def record(self, table: str, rid: str, sigma: float = float("inf"), viewer: str | None = None):
        return self.backend.record(self.materialize(sigma, viewer=viewer), table, rid)

    def log(self, **ev):
        self.events.append(ev)

    def view(self, tasks: set[str]):
        """Read-only projection: base + only the ops of the given workload tasks (for pair invariants)."""
        return _View(self, tasks)


class _View:
    def __init__(self, store: ObjectStore, tasks: set[str]):
        self.store, self.tasks, self.backend, self.base = store, tasks, store.backend, store.base

    def _ops(self):
        return [o for o in self.store.live_ops() if o.task in self.tasks]

    def materialize(self, sigma=float("inf"), exclude=None, viewer=None):
        with self.store.lock:
            st = self.backend.copy(self.base)
            for o in sorted(self._ops(), key=lambda o: (o.sigma, o.seq)):
                self.store._run(o, st, replaying=True)
            return st

    def read(self, tool, kwargs, sigma=float("inf"), exclude=None, viewer=None):
        with self.store.lock:
            st = self.materialize()
            op = Op(seq=-1, sigma=sigma, agent="?", tool=tool, kwargs=kwargs, store=self.store)
            return self.store._run(op, st, replaying=False)

    def record(self, table, rid, sigma=float("inf"), viewer=None):
        return self.backend.record(self.materialize(), table, rid)
