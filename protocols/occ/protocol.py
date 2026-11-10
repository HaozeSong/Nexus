"""OCC with eager validation and in-place writes.

Following the CoAgent baseline: "OCC reuses the same bindings under eager validation; the first rw/ww conflict commits the
trigger and aborts the conflicting agent, which restarts."  Writes land in place (there is no private
buffer on a live system); when agent X's write intersects the footprint of an in-flight agent Y,
X is the trigger and keeps its work, Y is marked doomed and aborted at its next call or at commit.
"""
from __future__ import annotations

from nexus.transaction import AbortAgent, AgentCtx, Protocol
from nexus.tools import ToolSpec, footprint_conflict


class P(Protocol):
    name = "occ"

    def __init__(self, store, agents):
        super().__init__(store, agents)
        self.doomed: dict[str, str] = {}

    def before(self, ctx: AgentCtx, spec: ToolSpec, kwargs: dict):
        if ctx.name in self.doomed:
            raise AbortAgent(self.doomed.pop(ctx.name))

    def after(self, ctx, spec, kwargs, result, op):
        if op is None:
            return
        for o in self.others(ctx):
            if o.committed or o.done or o.name in self.doomed:
                continue
            hit = footprint_conflict(op.write_set, o.read_set | o.write_set)
            if hit:
                self.doomed[o.name] = f"validation failed: {ctx.name} wrote {sorted(hit)} that you read/wrote"
                self.trace.append({"agent": o.name, "event": "doomed_by", "trigger": ctx.name, "objs": sorted(hit)})

    def validate(self, ctx):
        if ctx.name in self.doomed:
            raise AbortAgent(self.doomed.pop(ctx.name))

    def on_abort(self, ctx):
        self.doomed.pop(ctx.name, None)
