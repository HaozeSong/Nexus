"""Cordon: task-level semantic transactions (arXiv:2606.17573), realised on the shared harness.

In Cordon, a task's mutations execute against a transaction-scoped *shadow state*: "Subsequent reads
consult this view, so the agent can observe its speculative work while uncommitted changes remain
invisible outside the transaction."  External effects are staged in an *effect outbox*: "Each e in E remains
staged until validation succeeds and the transaction commits"; on abort the runtime "cancels entries that
remain pending".  At commit the engine "promotes approved writes and deletes to the real workspace".
Commit validation checks Cordon's nine invariants (secret flow, authority, policy, pre-commit staging,
rollback); it does not validate reads against concurrent tasks' commits, and its authors state that
cross-task interaction is not addressed.  Hence on contended cells Cordon behaves as shadow-buffered
last-writer-wins: it contains effects but does not isolate agents.  (MemTX's third-party reimplementation,
third_party/MEMTX_/src/baselines/cordon_manager.py, has exactly these semantics.)

Mapping: each agent's writes are ops flagged private until commit (shadow view = committed ops + own
private ops); unrecoverable tools (send_email) are held in the outbox and released at commit; nothing is
rejected at commit because the security invariants are vacuous for the WorkBench tasks.
"""
from __future__ import annotations

from nexus.transaction import AgentCtx, Protocol


class P(Protocol):
    name = "cordon"
    SHADOW_WRITES = True

    def __init__(self, store, agents):
        super().__init__(store, agents)
        self.outbox: dict[str, list] = {a: [] for a in agents}   # deferred external effects per task

    def read_exclude(self, ctx: AgentCtx):
        # shadow view: everyone else's uncommitted ops are private (invisible) until on_commit publishes them; nothing else is
        # excluded, so the committed earlier tasks of a still-running agent stay visible in multi-task workloads
        return None

    def call(self, agent, tool, kwargs):
        spec = self.store.backend.specs[tool]
        if spec.kind == "write" and spec.unrecoverable:
            ctx = self.agents[agent]
            with self.cv:
                self.outbox[agent].append((tool, kwargs))
                self.stats["holds"] += 1
                notes, ctx.pending = ctx.pending, []
            from nexus.transaction import CallResult
            return CallResult("Staged in the effect outbox; it will be released when your task commits.", notes, 0.0)
        return super().call(agent, tool, kwargs)

    def on_commit(self, ctx: AgentCtx):
        # promote shadow writes to the real workspace, then release the outbox in order
        for op in self.store.ops_of(ctx.name):
            op.private = False
        for tool, kwargs in self.outbox[ctx.name]:
            self.store.write(ctx.name, tool, kwargs)
        self.outbox[ctx.name] = []

    def on_abort(self, ctx: AgentCtx):
        self.outbox[ctx.name] = []    # pending entries cancelled; nothing crossed the boundary
