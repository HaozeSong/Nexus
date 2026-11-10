# S-Bus / ORI (rule ported from the official artifact)
Official artifact: https://github.com/sajjadanwar0/sbus (Rust: sbus-server, sbus-baselines, sbus-proxy) and
https://github.com/sajjadanwar0/sbus-experiments (Python harness). Their server is an HTTP middleware over
"shards" with `/shard` GET (logged into the DeliveryLog) and `/commit` (rejected with 409 when a cross-shard
read has been superseded).

The ORI rule is ported into the shared harness so every protocol sees the same
tools, agent loop and invariants: read set = observed footprint of each read (automatic, no declaration by
the agent), commit-time validation against other agents' writes, 409 → rollback + restart with the stale
reads named. The official `sbus-server` is not run.
