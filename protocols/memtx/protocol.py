"""MemTX baseline: WorkBench writes as belief records committed through the official `MemTXManager`.

Official code: third_party/MEMTX_/src/memtx (unmodified).  Per agent task: `begin_tx` (snapshot of
visible beliefs), every write `stage_write`s one MemoryRecord (entity="table:id", attribute=field, or
attribute="row" for create/delete) and stays a shadow op; irreversible tools go through
`action_safe_check` (blocked while other agents have TENTATIVE records in flight) and are recorded with
`record_tool_effect`; at task end `validate_and_commit` decides per record: committed records are
published, records aborted by the conflict detector (temporal_override = same slot committed by another
agent after this snapshot, or entity_attribute/ask_user) are rolled back and the agent is re-opened with
the list of rejected writes (typed repair: redo only those).  A repair write carries slightly higher
source authority so that MemTX's detector resolves it as `supersede_old` instead of `ask_user`.
"""
from __future__ import annotations

from nexus.paths import vendored

vendored("MEMTX_", "src")

import mock_tools  # noqa: E402
from memtx.memtx_manager import MemTXManager  # noqa: E402

from nexus.transaction import AgentCtx, Protocol  # noqa: E402
from nexus.tools import ToolSpec  # noqa: E402



class P(Protocol):
    name = "memtx"
    SHADOW_WRITES = True

    def __init__(self, store, agents):
        super().__init__(store, agents)
        mock_tools.IRREVERSIBLE_TOOLS |= {n for n, s in store.backend.specs.items() if s.unrecoverable}
        self.m = MemTXManager()
        self.tx = {a: self.m.begin_tx(task_id=a, agent_id=a, risk_level="medium") for a in agents}
        self.rec_op: dict[str, object] = {}
        self.rounds = {a: 0 for a in agents}   # repair rounds -> authority bump

    def before(self, ctx: AgentCtx, spec: ToolSpec, kwargs):
        if not spec.unrecoverable:
            return
        waited = False
        while True:
            gate = self.m.action_safe_check(self.tx[ctx.name], spec.name)
            if gate["allow"] or all(o.committed or o.done for o in self.others(ctx)):
                break
            waited = True
            self.cv.wait(timeout=0.5)
        if waited:
            self.stats["holds"] += 1
            self.trace.append({"agent": ctx.name, "event": "action_safe_released", "tool": spec.name})

    def after(self, ctx: AgentCtx, spec: ToolSpec, kwargs, result, op):
        if op is None:
            return
        tx = self.tx[ctx.name]
        if spec.unrecoverable:
            op.private = False                 # tool effects execute live in MemTX (gated beforehand)
            self.m.record_tool_effect(tx, spec.name, kwargs, {"result": result})
            return
        if spec.id_param and kwargs.get(spec.id_param):
            entity, attr = f"{spec.table}:{kwargs[spec.id_param]}", kwargs.get("field", "row")
        else:
            entity, attr = f"{spec.table}:{op.alloc_ids[0] if op.alloc_ids else '*'}", "row"
        mid = f"m_{ctx.name}_{op.seq}"
        rec = {"memory_id": mid, "entity": entity, "attribute": attr,
               "value": kwargs.get("new_value", op.brief()),
               "source": {"type": "agent", "id": ctx.name, "authority": 0.8 + 0.01 * self.rounds[ctx.name]},
               "confidence": 0.9, "derived_from": []}
        self.m.stage_write(tx, {"content": op.brief()}, [rec])
        self.rec_op[mid] = op

    def validate(self, ctx: AgentCtx):
        res = self.m.validate_and_commit(self.tx[ctx.name])
        for mid in res.get("committed_ids", []):
            self.rec_op[mid].private = False
        rejected = []
        for mid in res.get("aborted_ids", []):
            op = self.rec_op[mid]
            op.undone = True
            self.stats["undone_ops"] += 1
            rejected.append((op.brief(), [v for v in res["violations"] if v.endswith(mid)]))
        self.trace.append({"agent": ctx.name, "event": "memtx_commit", **{k: res.get(k) for k in ("status", "violations")}})
        self.tx[ctx.name] = self.m.begin_tx(task_id=ctx.name, agent_id=ctx.name, risk_level="medium")
        if rejected:
            self.rounds[ctx.name] += 1
            self.stats["aborts"] += 1
            self.notify(ctx, "MemTX rejected these write(s) at commit and rolled them back, because another agent "
                            "committed a newer value for the same record field after your snapshot: " +
                            "; ".join(f"{b} [{', '.join(v)}]" for b, v in rejected) +
                            ". Re-read the current values and redo only these writes if they are still needed.")

    def on_abort(self, ctx: AgentCtx):
        self.m.abort_or_rollback(self.tx[ctx.name], "agent-restart")
        self.tx[ctx.name] = self.m.begin_tx(task_id=ctx.name, agent_id=ctx.name, risk_level="medium")
