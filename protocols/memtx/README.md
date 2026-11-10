# MemTX (official code, adapted)
Uses `third_party/MEMTX_/src/memtx` unmodified through the `MemoryManager` interface (`begin_tx`,
`stage_write`, `validate_and_commit`, `action_safe_check`, `record_tool_effect`, `abort_or_rollback`).
Each WorkBench write is one belief record on slot (entity=`table:id`, attribute=field). The conflict
detector's concurrency rule is per-slot: a slot committed by another agent after this transaction's
snapshot → `temporal_override` → that record is aborted (partial commit), and the agent is re-opened with
the rejected writes to redo (typed repair). Different attributes of the same record, or different
records, do not conflict, so write skew and canary-style stale premises are not detected by design.
Dependency stability only tracks *revoked* ancestors, not superseded ones. Irreversible tools are gated
by `action_safe_check` while other agents hold TENTATIVE (uncommitted) records.
