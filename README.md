# Nexus

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Serializable transactions for concurrent LLM agents**, with conflict repair that reuses the inference a task has already done.

## Overview

LLM agents increasingly run concurrently over shared state: coding agents edit one repository, customer-service agents update
the same reservations. Interleaving their tool calls produces lost updates, stale reads and write skew. Classical concurrency
control prevents these by holding locks across inference or by discarding inference on abort, and both are expensive when a
task spends most of its time generating. **Nexus** is a transaction layer between the agent loop, the inference engine and the
shared state. It runs each agent task as a transaction and, when a concurrent commit invalidates something the task has read,
re-executes the task from that read against the updated state, using the task's previous output as a draft that the engine
verifies in parallel instead of decoding again.

**Key features:**
- Trajectory serializability: the context, tool calls and writes of every committed task match a serial execution in commit order
- Anchored replay: rollback to the earliest stale read, draft verification by the engine, new tokens only where the replay diverges
- Forward validation: a commit cancels the in-flight generations whose reads it invalidates
- Write intents: a reader waits for an older pending writer or proceeds speculatively, by expected repair cost and engine load
- Content-keyed sampling: an SGLang sampling backend that keeps the randomness of a replay aligned with the original execution
- Nine baseline protocols on the same harness (2PL, OCC, MTPO, S-Bus, Atomix, Cordon, MemTX, serial, naive)
- Agent workloads with correctness oracles: airline reservations (τ²-bench), a shared code repository, TPC-C, WorkBench

## Installation

```bash
pip install -e .
```

Nexus needs Python 3.12 or later and is used in place (`pip install -e .`), because it locates `third_party/` and `results/`
relative to the checkout. Inference runs on [SGLang](https://github.com/sgl-project/sglang) 0.5.x, installed separately on the
serving machine, with a model that supports tool calling.

The τ² and WorkBench workloads and two of the baselines reuse their authors' repositories:

```bash
# tau2-bench and WorkBench (workloads), atomix and MEMTX_ (baselines), at pinned commits
bash scripts/fetch_third_party.sh
pip install -e third_party/tau2-bench

# Optional: public agent traces and duckdb for the workload characterization script
bash scripts/fetch_third_party.sh traces
pip install -e ".[profiling]"
```

A workload whose repository is absent is skipped; the `workspace`, `micro` and `tpcc` workloads need none.

## How it works and defaults

Every task runs against a snapshot of the committed state overlaid with its own buffered writes, which stay private until the
task commits. Tasks are serialized in commit order.

1. **Versioned read sets.** Each tool declares the records it reads and writes. When a tool result enters the context, Nexus
   records its position and the versions of the records it observed.
2. **Forward validation.** When a task commits, its write set is compared with the read sets of the running tasks. A task that
   read an overwritten record is marked stale at its earliest such read, and its in-flight generation is cancelled in the engine.
3. **Anchored replay.** The stale task keeps the prefix before that read, retracts the buffered writes and intents issued after
   it, re-executes the read on the current state, and continues with its previous output as a draft. The engine verifies up to
   `VERIFY_CHUNK` draft tokens per forward pass, accepts the longest prefix that matches what it would sample under the updated
   context, emits its own token at the first mismatch, and re-aligns with the draft afterwards. Tool calls whose arguments and
   read versions are unchanged reuse their recorded results.
4. **Content-keyed sampling.** Sampling noise is keyed by the last `KEY_WINDOW` tokens and their occurrence count instead of the
   absolute position, so a tool result of a different length does not change the noise of the text that follows it.
5. **Write intents.** A buffered write registers an intent on its records. A younger reader of such a record waits for the
   writer or proceeds on the committed value, whichever is expected to be cheaper given the history of intents, the cost of past
   repairs and the load of the engine. Waits point from younger to older tasks only, so they cannot deadlock.
6. **Commit.** The read set is validated and the buffered writes are published atomically.

Defaults: `KEY_WINDOW = 8` tokens and `VERIFY_CHUNK = 64` tokens; the cost-based waiting policy; sampling temperature 1.0 and
top-p 0.95. A task has a budget of 30 model turns (`--max-steps`) and five restarts after aborts, and a Nexus task whose commit
fails `MAX_REPAIRS` (3) times aborts; a task that exhausts a budget fails. Any class attribute of a protocol can be overridden
from the command line with `--set KEY=VALUE`. `docs/DESIGN.md` describes the mechanisms and their rules in detail.

## Quick Start

Serve a model with the Nexus sampling backend, then run eight concurrent agents on the shared-repository workload:

```bash
python -m nexus.engine.sglang_plugin.launch --model-path <model> --tp 2 --port 30001 --context-length 65536 \
    --tool-call-parser <parser> --reasoning-parser <parser>

python scripts/run_concurrency.py --base-url http://127.0.0.1:30001/v1 --model <served name> --tokenizer <model path> \
    --keyed-engine --benchmark workspace --protocols serial,occ,nexus --agents 8
```

The driver prints one line per run and a summary table per protocol: serializable completion rate (SCR), GPU seconds per task,
tokens per task, and job completion time (JCT). Records are appended to `results/conc_workspace_<model>/runs.jsonl`.

## Python API

Run a workload under a protocol:

```python
from benchmarks.registry import resolve
from nexus.engine.sglang_native import SGLangNative
from nexus.workload import build_schedule, run_workload
from protocols import load_protocol

cells = resolve(["workspace"])                    # conflicting task pairs of a workload
schedule = build_schedule(cells, fillers=[], n_agents=8, n_tasks=16, p_c=1.0, delta="empirical", seed=0)


def engine(agent):
    return SGLangNative("http://127.0.0.1:30001/v1", "<served name>", "<model path>", keyed_engine=True)


run = run_workload(schedule, load_protocol("nexus"), cells[0].backend, engine)

print(f"SCR {run['pair_correct']}/{run['pairs']}")                   # pairs that completed serializably
print(f"JCT p50 {run['latency_p50']}s, p95 {run['latency_p95']}s")
print(run["stats"]["invalidations"], run["tokens_total"])            # protocol statistics, processed tokens
```

Run the two tasks of a single conflict cell:

```python
from nexus.runner import run_trial

cell = resolve(["tpcc/restock_then_half_order"])[0]
record = run_trial(cell, "nexus", {"A": engine("A"), "B": engine("B")}, trial=0)
print(record["correct"], record["detail"])
```

## CLI

All drivers share the engine arguments and resume from their result files, so an interrupted sweep is restarted with the same
command.

```bash
E="--base-url http://127.0.0.1:30001/v1 --model <served name> --tokenizer <model path> --keyed-engine"

# Concurrency sweep: N agents in a closed loop, every task in a conflicting pair
python scripts/run_concurrency.py $E --benchmark tau2long --protocols serial,twopl,occ,nexus_undo,mtpo,nexus --agents 2,4,8,16 --seeds 3

# Longer transactions: L tasks chained into one
python scripts/run_concurrency.py $E --benchmark workspace --agents 2,4,8 --long 1,2,3,4 --max-steps 64 --tasks 12 --tasks-per-agent 0

# Repair mechanisms and the sensitivity of replay to its parameters
python scripts/run_replay_study.py $E --benchmarks tau2long,workspace,micro --repairs nexus,nexus_nodraft,nexus_undo,nexus_restart
python scripts/run_replay_study.py $E --benchmarks tau2long --repairs "" --windows 1,2,4,8,16,32 --chunks 8,16,32,64,128,256

# Short conflict cells: two agents, one conflict per cell, every protocol
python scripts/run_short.py $E --cells workbench,tpcc,tau2 --per-template 10

# KV-cache pressure, reuse of cancelled output, overhead of the transaction layer
python scripts/run_engine.py kv $E --servers 102k=http://HOST:30001/v1,32k=http://HOST:30003/v1 --benchmark workspace

# Figures from the result files
python scripts/make_figures.py
```

`docs/EXPERIMENTS.md` lists every experiment with its command and outputs.

| engine argument | meaning |
|---|---|
| `--engine sglang` (default) | token-level SGLang client: drafts, verification, cancellation, keyed sampling |
| `--engine openai` | any OpenAI-compatible server: cancellation only, so replay regenerates without drafts |
| `--keyed-engine` | the server was started through `nexus.engine.sglang_plugin.launch` |
| `--tool-format` | `minimax` (default), `hermes`, or `json` |
| `--temperature`, `--top-p`, `--max-tokens` | sampling parameters (1.0, 0.95, 8192) |

Protocols, selected by name (`--protocols`):

| name | mechanism |
|---|---|
| `nexus` | versioned read sets, forward validation, anchored replay, write intents |
| `serial` | tasks run back to back: the reference for correctness and cost |
| `naive` | no coordination |
| `twopl` | strict two-phase locking with range locks, deadlock detection and saga rollback |
| `occ` | optimistic execution with eager validation; the conflicting task aborts and restarts |
| `mtpo` | monotonic trajectory pre-order (CoAgent): speculative writes, undo and notification |
| `sbus` | per-record versions, commits rejected on stale reads |
| `atomix` | saga and compensation runtime (vendored) |
| `cordon` | transaction-scoped shadow state and an effect outbox |
| `memtx` | memory transactions (vendored) |

Variants of Nexus change one mechanism each: the repair (`nexus_undo`, `nexus_nodraft`, `nexus_restart`, `nexus_notify`), the
waiting policy (`nexus_wait`, `nexus_proceed`, `nexus_nointents`), engine priority (`nexus_nopriority`), sampling keys
(`nexus_poskeys`) and cancellation (`nexus_nocancel`).

Workloads, selected by suite name (`--benchmark`, `--cells`) or as `benchmark/cell`:

| suite | tasks |
|---|---|
| `tau2long` | airline customer cases on one reservation: flight changes, cabin, baggage, passengers, cancellations |
| `workspace` | a shared Python repository: implement functions until the tests pass, then update the changelog and the version |
| `micro` | TPC-C inventory and credit reviews with controlled read/write-set sizes and read/write positions |
| `tpcclong` | analyse-then-act procedures over several TPC-C rows |
| `workbench`, `tpcc`, `tau2` | short two-agent conflict cells: lost update, write skew, phantom, stale read before an irreversible action |

## Notes

- **Exact verification**: with the sampling backend (`--keyed-engine`) draft verification is exact at any temperature. Without it
  the client still verifies drafts with keyed sampling, which is exact at temperature 0 and approximate otherwise.
- **Correctness**: a pair of concurrent tasks is correct when the final state equals the outcome of one of the two serial
  executions of the tasks' reference procedures and both tasks finish within their budget.
- **GPU time**: engine time is attributed to requests from the server's `/metrics`, and includes aborted and cancelled work.
- **Vendored data**: `TAU2_DATA_DIR` defaults to `third_party/tau2-bench/data`.
- **Extending**: a protocol is a `Protocol` subclass named `P` in `protocols/<name>/protocol.py` (`nexus/transaction.py` has the
  hooks); a workload is `benchmarks/<name>/{backend,cells}.py`, registered in `benchmarks/registry.py`. A backend declares the
  access set of each tool (`nexus/tools.py`).

## Project structure

```
nexus/         the transaction layer: store, read sets, intents, manager, replay, sampling; agent loop and workload driver
nexus/engine/  inference engines (SGLang, OpenAI-compatible, simulated) and the SGLang sampling backend
protocols/     one package per protocol; the Nexus variants are in protocols/nexus/ablations.py
benchmarks/    one package per workload; registry.py lists the suites
scripts/       experiment drivers, figures, third-party setup
docs/          DESIGN.md (mechanisms), EXPERIMENTS.md (experiments)
tests/         unit tests
```

## Tests

```bash
# Sampling semantics, anchored replay on the simulated engine, protocol matrix (no GPU needed)
python -m unittest discover -s tests -v

# Every protocol on the conflict cells of a suite, with scripted agents
python scripts/scripted_test.py serial,twopl,occ,mtpo,nexus tpcc
```

## License

MIT. The vendored repositories keep their own licenses.
