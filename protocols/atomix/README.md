# Atomix (official code, adapted)
Uses `third_party/atomix` unmodified: `OCCRevalidateRetry` (the repository's multi-agent CC baseline,
used in its E2 multi-agent experiments), `Effect`, `Epoch`. The core Atomix runtime's frontier/epoch
machinery orders effects within one trace and provides compensation and irreversible-effect gating; it
has no cross-agent conflict detection, which is why the repository ships OCC/mutex baselines for that.
Adapter notes: task-granularity transaction per agent; read stamps from observed footprints; predicate
reads mapped by also bumping the `table:*` version on every write; stale read → agent restart (their
"retry"), budget 3; irreversible effects confirmed and applied at commit (deferred, never leaked).
