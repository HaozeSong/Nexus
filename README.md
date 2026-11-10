# Nexus

Nexus is a transaction-management layer for concurrent LLM agents that share mutable state through tools. This repository is the
artifact of the paper *Nexus: Making Concurrent LLM Agents Serializable and Efficient with Anchored Replay*: the transaction layer
(versioned read sets, forward validation, anchored replay with content-keyed sampling, write intents), the baselines, the
workloads, and the drivers of every experiment. Sections 2 and 3 map the paper's protocols and workloads to their names here;
`docs/EXPERIMENTS.md` maps every figure and table to its command.

| paper | code |
|---|---|
| §III-A execution model, trajectory serializability | `nexus/store.py`, `nexus/transaction.py`, `nexus/agent.py`; `docs/DESIGN.md` |
| §III-B snapshots and versioned read sets | `nexus/readsets.py`, `nexus/tools.py` (access sets), `nexus/manager.py` |
| §III-C forward validation | `nexus/manager.py` (cancellation through `nexus/engine/`) |
| §III-D anchored replay (Algorithm 1) | `nexus/replay.py`, `nexus/agent.py`; verify/generate and Align in `nexus/sampling.py` |
| §III-E content-keyed sampling | `nexus/sampling.py`, `nexus/engine/sglang_plugin/` |
| §III-F write intents | `nexus/intents.py`, `nexus/manager.py` |
| §IV evaluation | `scripts/`, `docs/EXPERIMENTS.md` |

## Quickstart: run the protocol matrix without a model

```bash
python3.12 -m venv .venv && . .venv/bin/activate && pip install -e .       # installs nexus/, protocols/, benchmarks/ in place
bash scripts/fetch_third_party.sh                       # tau2-bench, WorkBench (benchmarks); atomix, MEMTX_ (baselines)
pip install -e third_party/tau2-bench                   # only for the tau2 suites; a suite whose repository is absent is skipped
python -m unittest discover -s tests -v                 # sampling semantics, anchored replay on the simulated engine, protocol matrix
python scripts/scripted_test.py serial,naive,twopl,occ,mtpo,nexus tpcc      # every TPC-C conflict cell under six protocols
```

All commands run from the repository root. The scripts import the installed packages, so `pip install -e .` (or the equivalent
`pip install -r requirements.txt`) is required once; the repository is used in place and is not copied.


## 1. Directory Layout

```
nexus/                      the system
  store.py                  multi-version log store: committed vs buffered ops, per-task snapshots
  tools.py                  access sets from tool schemas (table:id records, table:* listings, table:+ inserts), conflict relation
  readsets.py               versioned read sets and the staleness tests used by validation, premise checks and replay
  intents.py                write intents: age-ordered waits, early invalidation of younger readers, the wait/proceed pricing rule
  manager.py                NexusTM: snapshots, forward validation with engine-side cancellation, premise checks, commit, intents
  replay.py                 anchored replay on the trajectory: rollback to the earliest stale read, drafts, memoized tool calls
  sampling.py               content-keyed sampling: PRF, (window, occurrence) keys, Gumbel-max sampling, Align, verify/generate
  engine/                   inference engines and the SGLang sampler plugin
  -- harness shared by every protocol --
  transaction.py            Protocol: the tool-call middleware interface every protocol implements
  speculative.py            speculative-write substrate (in-place writes, undo, notifications) used by MTPO
  backend.py / cell.py      what a benchmark provides: a Backend (state + tools) and Cells (two tasks + a correctness oracle)
  agent.py                  the agent loop: turns, tool calls, repairs (replay | regen | undo | notify | restart)
  runner.py / serial.py     two-agent trials, metrics, GPU-time attribution; serializable-outcome oracles
  workload.py               closed-loop workload driver: N agents, contended pairs, release points, chained transactions
  paths.py                  repository root and the vendored source trees under third_party/
protocols/                  one package per protocol (<name>/protocol.py); nexus/ablations.py holds the Nexus ablations
benchmarks/                 one package per benchmark (tau2, workspace, micro, tpcc, workbench); registry.py lists the suites
scripts/
  run_concurrency.py  run_replay_study.py  run_short.py  run_engine.py  occupancy_sweep.py     model-backed experiment drivers
  scripted_test.py  workload_test.py                                                          no-model drivers (scripted agents)
  make_figures.py  profile_workloads.py  reeval_workspace.py                                  analysis of result files
  fetch_third_party.sh  _common.py                                                            setup; CLI pieces shared by the drivers
tests/                      unit tests
docs/                       DESIGN.md (how the pieces fit), EXPERIMENTS.md (every experiment, end to end)
third_party/                vendored benchmark/baseline repositories (fetched by scripts/fetch_third_party.sh, not tracked)
results/                    scripted_matrix_full.txt: the no-model protocol matrix
```

## 2. Choosing a Protocol

Every protocol is a `Protocol` subclass with the same interface (`before`/`call`/`after` around each tool call, `commit`,
`abort`), selected by name in every script (`protocols.load_protocol(name)`; `protocols.available()` lists the names). The
protocols of the table below live in `protocols/<name>/protocol.py`; each baseline has a README on how it maps onto the harness.

| name | in the paper | mechanism | outcome on a conflict |
|---|---|---|---|
| `serial` | Serial | tasks run back to back | reference for correctness and cost |
| `naive` | Naive | no coordination | anomalies (lost updates, write skew) |
| `twopl` | 2PL | strict two-phase locking with range locks, deadlock detection, saga rollback | victim aborted and restarted |
| `occ` | OCC | eager optimistic execution, validation at every write | first conflicting task aborted |
| `mtpo` | MTPO (CoAgent) | monotonic trajectory pre-order: speculative writes, undo + notification | later task undoes and reconciles |
| `sbus` | S-Bus | serialized tool bus with per-record versions (409 on stale writes) | rejected write, agent retries |
| `atomix` | Atomix | saga/compensation runtime (vendored) | compensating actions |
| `cordon` | Cordon | prompt-level guards and transaction-scoped shadow state | no read validation |
| `memtx` | MemTX | memory transactions (vendored) | no read validation |
| `nexus` | NEXUS | versioned read sets, forward validation, anchored replay, write intents | invalidated task replays from the stale read |
| `nexus_undo` | Undo | Nexus with selective undo instead of replay: stale results refreshed in place, the model redoes the affected steps | affected steps regenerated |

Ablations of Nexus are separate protocols so that every experiment selects them by name: repair mechanism (`nexus_restart` =
Restart, `nexus_nodraft` = regeneration without drafts, `nexus_undo` = Undo, `nexus_notify`), intent policy (`nexus_wait` = always
wait, `nexus_proceed` = always proceed, `nexus_nointents`), engine priority (`nexus_nopriority`), sampling keys (`nexus_poskeys` =
position keys), cancellation (`nexus_nocancel`). Each is a few class attributes over `nexus.manager.NexusTM`, all defined in
`protocols/nexus/ablations.py`; `scripts/run_concurrency.py --set KEY=VALUE` overrides any attribute (e.g. `KEY_WINDOW` = m,
`VERIFY_CHUNK` = b; the defaults are the paper's m = 8 and b = 64).

## 3. Workloads

| suite | in the paper | module | tasks | what it stresses |
|---|---|---|---|---|
| `tau2long` | τ² | `benchmarks/tau2/long.py` | airline reservations (τ²-bench): multi-request customer cases on one reservation (flight changes, cabin, baggage, passengers, cancellations) | long transactions whose writes depend on earlier reads |
| `workspace` | Workspace | `benchmarks/workspace/cells.py` | a shared Python repository (12 modules × 4 functions) edited through file tools: implement functions until the module tests pass, then add a `CHANGELOG.md` entry and bump the version | iterative edit–test loops that end in read-modify-write updates of metadata shared by every task |
| `micro` | Micro. | `benchmarks/micro/cells.py` | TPC-C inventory and credit reviews with controlled read/write-set sizes and read/write positions | where in a task an invalidation lands |
| `tpcclong` | — | `benchmarks/tpcc/long.py` | analyse-then-act procedures over several TPC-C rows (restock sweeps, bulk orders, delivery and credit reviews) | secondary long-transaction workload |
| `workbench`, `tpcc`, `tau2` | WorkBench, Micro-short, τ²-short | `benchmarks/<name>/cells.py` | short two-agent conflict cells on the WorkBench office tools (calendar, e-mail, CRM, project management), the TPC-C database and the τ² airline tools (lost update, write skew, phantom, cancel vs deliver, ...) | one anomaly per cell |

The short suites instantiate each conflict template on several disjoint record sets (`workbench`: 48 cells from 7 templates; `tpcc`:
5 templates × 18; `tau2`: 5 templates × 40). The paper's short-conflict table uses the 48 WorkBench cells and the first 10
instances per template of the other two (50 cells each): `scripts/run_short.py --per-template 10`.

A suite is one module that exports `CELLS`; `benchmarks/registry.py` maps suite names to modules, and every script selects cells
by suite name, by `benchmark/cell`, or by bare cell name. Every benchmark package has `cells.py` (tasks with a correctness oracle)
and `backend.py` (the tools, with declared access sets; `micro` reuses the TPC-C backend); `tau2` and `tpcc` add `long.py` for the
long-transaction suite on the same backend, and `tau2`, `tpcc`, `workbench` and `micro` have `scripted.py` (a deterministic
no-model agent per task). `tau2long`, `micro` and `tpcclong` are long-transaction workloads whose oracle is the serializable
outcome of reference procedures (`nexus/serial.py`); `workspace` requires both tasks' tests to pass and both changelog entries to
be present, and judges the shared changelog and version over the whole run (`run_metrics`); the short suites use hand-written
invariants that accept either serial order.

## 4. Correctness Semantics

A pair of concurrent tasks is **correct** when the final state equals the state produced by one of the two serial executions of
the tasks' reference procedures, compared through the backend's own tools. Every protocol is judged by this oracle under the same
harness, model, prompts and workload driver. Reported metrics are the serializable completion rate (SCR: the fraction of pairs in
which both tasks complete within their budget and the pair is correct), GPU time per task (engine wall time attributed to running
requests from the server's telemetry, including failed, aborted and cancelled work), tokens per task (input, cached, output),
and job completion time (JCT).

The per-task budget is a step budget of model turns (`--max-steps`, 30 by default and 64 in the transaction-length experiment)
and five restarts after aborts (`Protocol.RETRY_CAP`; `atomix` keeps the retry budget of its original runtime, three); a Nexus
task whose commit fails `MAX_REPAIRS` times aborts. A run without any progress for ten minutes is stopped (`STALL_S`). A task
that exhausts a budget fails, and so does its pair.

## 5. Engine Interface

`nexus/engine/base.py` defines what the layer needs from an inference engine: a chat call that returns the turn (text, tool calls,
usage, and token ids when available), cancellation of an in-flight request, and optionally a **draft** (token ids of a previous
continuation) with **content-keyed sampling parameters**. Three engines implement it:

| engine | use |
|---|---|
| `sglang_native` | token-level client for SGLang: builds the prompt with the model's chat template, streams `/generate`, cancels through `/abort_request`, verifies draft chunks through the logprob interface, samples with content keys |
| `openai_chat` | any OpenAI-compatible server; cancellation only, no token ids, so replay degrades to regeneration without drafts |
| `sim` | simulated token-level engine around a scripted policy, used by the unit tests to exercise drafts, verification and cancellation without a model |

## 6. Running Without a Model

```bash
python -m unittest discover -s tests -v                      # 25 tests, about a minute
python scripts/scripted_test.py <protocols> <suite|cells>   # e.g. serial,nexus tpcc,micro or nexus tpcc/restock_then_half_order
python scripts/workload_test.py nexus,occ 4 8 1.0 after_read # closed-loop driver with scripted agents: N=4, 8 tasks, all contended
python scripts/run_engine.py overhead --agents 2,4,8,16      # wall time of the Nexus layer per tool call and per commit
```


## 7. Running With a Model

Start an SGLang server (with the sampler plugin, section 9, for exact draft verification under sampling) and run the drivers.
`docs/EXPERIMENTS.md` describes every experiment; the main ones:

```bash
python -m nexus.engine.sglang_plugin.launch --model-path <model> --tp 2 --port 30001 --tool-call-parser <parser> --reasoning-parser <parser>
E="--engine sglang --base-url http://127.0.0.1:30001/v1 --model <served name> --tokenizer <model path> --keyed-engine"

python scripts/run_concurrency.py $E --benchmark tau2long --protocols serial,twopl,occ,nexus_undo,mtpo,nexus --agents 2,4,8,16 --seeds 3
python scripts/run_replay_study.py $E --benchmarks tau2long,workspace,micro --repairs nexus,nexus_nodraft,nexus_undo,nexus_restart,nexus_notify
python scripts/run_short.py $E --cells workbench,tpcc,tau2 --per-template 10 --protocols serial,naive,twopl,occ,mtpo,sbus,atomix,cordon,memtx,nexus
python scripts/make_figures.py                               # results/figures/*.pdf
```

Every driver appends to `results/<experiment>/runs.jsonl` (one record per run with per-task latency, tokens, tool calls with
draft accounting, protocol statistics, pair verdicts and the operation log) and resumes from it, so interrupted sweeps are
restarted with the same command. `--engine openai` runs the same drivers against any OpenAI-compatible server.

## 8. Pre-computed Results

`results/scripted_matrix_full.txt` is the no-model matrix: scripted agents on every instance of the three short
suites (338 cells, a superset of the 148 cells of the paper's short-conflict table) x 13 protocols, produced by

```bash
python scripts/scripted_test.py serial,naive,twopl,occ,mtpo,sbus,atomix,cordon,memtx,nexus,nexus_nodraft,nexus_nointents,nexus_restart \
    tpcc,tau2,workbench > results/scripted_matrix_full.txt
```

| protocol | cells correct | protocol | cells correct |
|---|---|---|---|
| serial | 338 / 338 | atomix | 338 / 338 |
| naive | 42 / 338 | cordon | 43 / 338 |
| twopl | 338 / 338 | memtx | 230 / 338 |
| occ | 338 / 338 | nexus | 338 / 338 |
| mtpo | 338 / 338 | nexus_nodraft | 338 / 338 |
| sbus | 337 / 338 | nexus_restart | 338 / 338 |
| | | nexus_nointents | 333 / 338 |

`cordon` and `memtx` fail the lost-update cells by design (no read validation). The six other misses are timing races of the
scripted agents (one S-Bus cell, five instances of one τ² cell under the no-intents ablation) and pass when rerun on their own.
The model-backed results reported in the paper are regenerated by the drivers of section 7 (`docs/EXPERIMENTS.md` gives the
command for every figure and table); their raw runs are not included.

## 9. Engine Plugin: Content-Keyed Sampling in SGLang

SGLang's deterministic sampler draws its Gumbel noise from a hash of `(seed, position, token)`. The plugin in
`nexus/engine/sglang_plugin/` registers a sampling backend `nexus` whose noise is keyed by `(seed, content key, token)`, where the
content key is a hash of the last *m* token ids and the number of earlier occurrences of that window in the sequence. Requests opt
in through `sampling_params.custom_params`; other requests are unaffected. The same PRF is implemented in pure Python in
`nexus/sampling.py`, so the client-side verifier and the engine draw identical noise (`tests/test_sampling.py` checks the two
against each other). The plugin is written against SGLang 0.5.x, and the model-backed experiments run the server with it
(`--keyed-engine`). Without it, the client still verifies drafts with keyed sampling while ordinary decoding uses the engine's
sampler, which is exact at temperature 0 and approximate otherwise. See `nexus/engine/sglang_plugin/README.md`.

## 10. Adding a Protocol or a Benchmark

* **Protocol:** create `protocols/<name>/protocol.py` with a `Protocol` subclass named `P` (see `protocols/occ/protocol.py` for a
  compact example and `nexus/transaction.py` for the base class and hooks); the name is picked up automatically by every script.
* **Nexus variant:** subclass `nexus.manager.NexusTM` in `protocols/nexus/ablations.py`, set `name` and the class attributes that
  differ (`REPAIR`, `DRAFTS`, `INTENTS`, `INTENT_POLICY`, `PRIORITIES`, `PREEMPT`, `KEY_WINDOW`, `VERIFY_CHUNK`, `MAX_REPAIRS`),
  and append the class to `ABLATIONS`.
* **Benchmark:** add `benchmarks/<name>/{backend,cells,scripted}.py`. The backend declares each tool's access set
  (`nexus/tools.py`); long-transaction cells are built with `nexus.serial.serializable_cell` from two reference procedures and an
  extractor of the touched records; each cell names the records its tasks touch (`Cell.objects`) so that the workload driver can
  pair record-disjoint cells; register the module under a suite name in `benchmarks/registry.py` (`SUITES`). A benchmark that
  reuses a repository under `third_party/` makes it importable with `nexus.paths.vendored`.

## 11. Toolchain

* Python 3.12; `pyproject.toml` lists the packages (numpy, pandas, pydantic, requests, openai, matplotlib; transformers for
  the token-level client; the optional `profiling` extra adds duckdb for `scripts/profile_workloads.py`).
* Inference: SGLang 0.5.x with a model that supports tool calling (the harness has prompt builders for the MiniMax and Hermes
  tool-call formats and a generic JSON fallback, `nexus/engine/toolcalls.py`). The paper's experiments serve MiniMax-M2.7 and
  DeepSeek-V4-Flash on NVIDIA H200 GPUs with two-way tensor parallelism and a 64k context window; any server the client can reach
  works.
* Vendored repositories (`scripts/fetch_third_party.sh`): tau2-bench and WorkBench for the benchmarks, atomix and MEMTX_ for the
  two baselines that reuse their authors' runtimes. `TAU2_DATA_DIR` defaults to `third_party/tau2-bench/data` and only needs to
  be set when the τ² data lives elsewhere. `scripts/fetch_third_party.sh traces` additionally fetches the public TraceLab traces
  used by the workload characterization.

## 12. Citation

Anonymized for review.

## License

MIT (see `LICENSE`). The vendored repositories keep their own licenses.
