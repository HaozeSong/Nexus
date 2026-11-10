# Cordon (reimplemented from the Cordon description)
Semantics on the shared harness: shadow state (own uncommitted writes visible only to the task), effect
outbox (send_email/reply/forward deferred to commit, cancelled on abort), commit = promote shadow writes.
Cordon's commit validation enforces security invariants (I1–I9: secret flow, authority, policy, staging,
rollback), not isolation between concurrent tasks; the Cordon description explicitly leaves cross-task interaction out
of scope. On the contended cells it therefore behaves as buffered last-writer-wins: containment is not
concurrency control.
