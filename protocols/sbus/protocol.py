"""S-Bus / Observable-Read Isolation (arXiv:2605.17076), ported to the shared harness.

S-Bus records every read an agent performs (DeliveryLog) to reconstruct its read set automatically and,
at commit, rejects the commit (HTTP 409) if any read has been superseded by another agent's write.  Here
the read set comes from the observed footprint of each read tool (the ids returned by a search, or the
record fetched), and each read stores the object's version, i.e. how many other-agent writes had touched
it.  Reads are live (no filtering); writes land in place.  A rejected commit rolls the agent back and
restarts it with the list of stale reads (S-Bus returns the offending shards).
"""
from __future__ import annotations

from nexus.transaction import AbortAgent, AgentCtx, Protocol
from nexus.tools import ToolSpec, conflicts


class P(Protocol):
    name = "sbus"

    def __init__(self, store, agents):
        super().__init__(store, agents)
        self.read_versions: dict[str, dict[str, int]] = {a: {} for a in agents}

    def _version(self, agent: str, obj: str) -> int:
        return sum(1 for op in self.store.live_ops()
                   if op.agent != agent and any(conflicts(w, obj) for w in op.write_set))

    def after(self, ctx: AgentCtx, spec: ToolSpec, kwargs, result, op):
        if spec.kind != "read":
            return
        r, _ = spec.observed(kwargs, result, [])
        rv = self.read_versions[ctx.name]
        for obj in r:
            rv.setdefault(obj, self._version(ctx.name, obj))

    def validate(self, ctx: AgentCtx):
        stale = [obj for obj, v in self.read_versions[ctx.name].items() if self._version(ctx.name, obj) != v]
        if stale:
            self.trace.append({"agent": ctx.name, "event": "commit_rejected_409", "stale": stale})
            raise AbortAgent(f"commit rejected: reads superseded on {sorted(stale)}")

    def on_abort(self, ctx):
        self.read_versions[ctx.name] = {}
