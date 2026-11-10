"""Atomix baseline: the repository's own multi-agent CC (`atomix.baselines.OCCRevalidateRetry`) driven
through Atomix `Effect`/`Epoch` objects, one Atomix transaction per agent task.

What the code does (third_party/atomix/src/atomix/baselines/occ_revalidate_retry.py): reads stamp
(scope, version); writes are recorded as effects and applied only at commit; commit revalidates every
stamp under a lock, bumps versions of the write set, and returns False on a stale read (the caller must
redo and re-commit); after `retry_budget` (3) stale attempts the transaction is permanently aborted.
Irreversible effects must be `confirmed` before commit, otherwise commit vetoes.

Mapping onto the harness: writes are shadow ops (private until Atomix applies the effect, which
publishes the op); each read stamps its observed footprint; a write on "table:id" also bumps "table:*"
so that range (predicate) readers are invalidated, since Atomix versions are keyed by exact resource
string; irreversible tools (send_email) are confirmed at commit time, i.e. released with the commit and
never leaked on abort; a stale-read result restarts the agent (its way of "redoing" the transaction).
"""
from __future__ import annotations

from nexus.paths import vendored

vendored("atomix", "src")

from atomix.baselines.occ_revalidate_retry import OCCRevalidateRetry, _ReadStamp  # noqa: E402
from atomix.effects import Effect, EffectReversibility  # noqa: E402
from atomix.epoch import Epoch  # noqa: E402

from nexus.transaction import AbortAgent, AgentCtx, Protocol  # noqa: E402
from nexus.tools import ToolSpec  # noqa: E402


class P(Protocol):
    name = "atomix"
    SHADOW_WRITES = True
    RETRY_CAP = 3          # Atomix's retry_budget

    def __init__(self, store, agents):
        super().__init__(store, agents)
        self.mgr = OCCRevalidateRetry(apply_effect=self._apply, retry_budget=3)
        self._epoch = 0
        self.tx = {a: self._begin(a) for a in agents}

    def _begin(self, agent):
        self._epoch += 1
        return self.mgr.begin(set(), Epoch(self._epoch, "trial", agent))

    def _apply(self, effect: Effect):
        effect.payload["op"].private = False      # publish the shadow write

    def after(self, ctx: AgentCtx, spec: ToolSpec, kwargs, result, op):
        tx = self.tx[ctx.name]
        if op is None:
            r, _ = spec.observed(kwargs, result, [])
            st = self.mgr._states[tx.tx_id]
            with self.mgr._versions_lock:
                for obj in r:
                    st.read_set.append(_ReadStamp(scope=obj, version_seen=self.mgr._versions[obj]))
            return
        scopes = set(op.write_set) | {f"{spec.table}:*"}
        eff = Effect(description=op.brief(), scopes=scopes, payload={"op": op},
                     idempotency_key=f"{ctx.name}:{op.seq}",
                     compensation=lambda op=op: setattr(op, "undone", True),
                     reversibility=EffectReversibility.IRREVERSIBLE if spec.unrecoverable else EffectReversibility.REVERSIBLE)
        self.mgr.record_effect(tx, eff)

    def validate(self, ctx: AgentCtx):
        tx = self.tx[ctx.name]
        for e in tx.effects:
            if e.reversibility == EffectReversibility.IRREVERSIBLE:
                e.confirmed = True        # settle irreversible effects with the commit
        ok = self.mgr.commit(tx)
        if ok:
            return
        self.trace.append({"agent": ctx.name, "event": "atomix_commit_failed", "reason": tx.reason, "status": tx.status})
        raise AbortAgent(f"atomix {tx.reason or tx.status}")

    def reset_agent(self, ctx: AgentCtx):
        self.tx[ctx.name] = self._begin(ctx.name)    # fresh Atomix transaction for the next task

    def on_abort(self, ctx: AgentCtx):
        old = self.tx[ctx.name]
        if old.status != "aborted":
            self.mgr.abort(old, "agent-restart")
        self.tx[ctx.name] = self._begin(ctx.name)
