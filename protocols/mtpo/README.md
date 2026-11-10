# MTPO / CoAgent (reimplemented from the paper; no public code)

How each rule of the published protocol is implemented on the shared harness:

| Rule in CoAgent (arXiv:2606.15376; § refers to that document) | Implementation here | Status |
|---|---|---|
| Pre-order: each agent gets rank σ at launch; notifications flow low→high only (§5.3) | σ = launch index; `after()` notifies only agents with higher σ | same |
| Filtered read: FilteredRead(j,o) = materialisation of the trajectory prefix with σ ≤ σ_j (§5.3, §6.2) | `store.materialize(σ_j)` = base + replay of ops with σ ≤ σ_j in (σ, seq) order; range reads materialise the whole table | same ("replay on a materialisation" route) |
| Speculative write: takes effect in place at its ≺t position and joins T(o) at rank σ_j (§5.3) | `store.write(sigma=σ_j)`: executed immediately, recorded with rank σ_j | same |
| Late write, blind entry above σ_j shadows w (Thomas write rule): w recorded, never replayed on the live copy | same-field blind writes (same tool, same record, same `field`) of the higher-σ agent are kept; the late write is recorded and, in replay order, shadowed | same |
| Late write, otherwise: undo the suffix above σ_j in descending σ using the tools' inverses, apply w, notify each undone agent which decides whether to retry (§5.3, §6.2) | `LATE_WRITE="undo"` (default): higher-σ ops on the conflicting objects are marked undone (the log-replay store is the inverse), w applied, undone agents notified with the list of undone actions | same |
| Notification ν_{i→j}(o): delivered at the receiver's next inference, carries refreshed M(o, σ_j); receiver may keep, rewrite, or retract (§5.1, §5.3, §6) | rides on the next tool return (or re-opens the agent after its final answer); carries the record as left by lower-σ agents and (if kept) as seen with own writes; `nexus.retract_my_actions` implements retract | same |
| Unrecoverable tools held until every lower-σ agent has committed (§6.3) | `before()` blocks unrecoverable tools until lower-σ agents committed | same |
| Commit hook: hold commit until pending notifications drain (GlobalQuiet, §6) | commit returns pending notifications (agent re-opened); additionally a higher-σ agent waits for lower-σ commits so no notification can arrive after its commit | same + stricter (no effect on final state; wall-clock of the pair is unchanged since the lower agent is still running) |
| Inverses registered per tool (three-phase toolcalls, §6.3) | not hand-written: dropping an op from the σ-ordered log and re-materialising is the inverse; unrecoverable tools declared per tool | equivalent for the sandboxed benchmarks |
| ToolSmith / object tree (§6.1, §6.4) | WorkBench/TPC-C/τ² tools enrolled by a one-time footprint declaration (CoAgent did the same for WorkBench, §7.4) | not needed |
| Retry cap 5 for 2PL/OCC (§7.1) | same | same |
