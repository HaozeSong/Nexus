# third_party

Vendored benchmark and baseline repositories. They are not tracked here; `scripts/fetch_third_party.sh` clones each at the
commit this artifact was developed against (`all` = benchmarks + baselines; `traces` is fetched separately):

| directory | group | used by | origin | commit |
|---|---|---|---|---|
| tau2-bench | benchmarks | benchmarks/tau2 (`TAU2_DATA_DIR` defaults to `third_party/tau2-bench/data`) | https://github.com/sierra-research/tau2-bench.git | `b7ea907` |
| WorkBench | benchmarks | benchmarks/workbench | https://github.com/olly-styles/WorkBench.git | `49c7dfd` |
| atomix | baselines | protocols/atomix (its original saga/compensation runtime) | https://github.com/mpi-dsg/atomix.git | `61e411b` |
| MEMTX_ | baselines | protocols/memtx (its original memory-transaction manager) | https://github.com/lxy1134/MEMTX_.git | `4e1124c` |
| TraceLab | traces | scripts/profile_workloads.py only (public agent traces for the workload characterization; the trace database is a release asset downloaded into `TraceLab/trace/`) | https://github.com/uw-syfi/TraceLab.git | `11b8b14` |

The other baselines (2PL, OCC, MTPO, S-Bus, Cordon) are reimplemented in `protocols/` from their published descriptions and have no vendored
dependency. Each repository keeps its own license. Code that imports from one of these trees makes it importable through
`nexus.paths.vendored` (the only place that edits `sys.path`).
