# Experiments

Every experiment below names the figure or table of the paper it produces. All commands run from the repository root after
`pip install -e .` and `bash scripts/fetch_third_party.sh` (see the README). `$E` stands for the engine arguments:

```bash
E="--engine sglang --base-url http://HOST:PORT/v1 --model <served name> --tokenizer <model path> --keyed-engine"
```

(`--keyed-engine` when the server was started through `nexus.engine.sglang_plugin.launch`; omit it, and exact verification holds
at temperature 0 only.) Every driver resumes from its `runs.jsonl` / `trials.jsonl`, so an interrupted sweep is restarted with the
same command. Results land under `results/<experiment>_<benchmark>_<model>/`.

| paper | experiment | driver |
|---|---|---|
| Fig. 3, Table I: workload characterization | E8 | `scripts/profile_workloads.py` |
| Fig. 8: end-to-end performance under increasing concurrency | E1 | `scripts/run_concurrency.py` |
| Fig. 9, Fig. 10: effectiveness of anchored replay, parameter sensitivity | E2 | `scripts/run_replay_study.py` |
| Fig. 11: longer transactions | E3 | `scripts/run_concurrency.py --long` |
| Table II: short-conflict workloads | E4 | `scripts/run_short.py` |
| Fig. 12, Table II (second model) | E7 | E1 and E4 with the second model |
| Fig. 13: cost breakdown and adaptive waiting | E1, E5 | `scripts/run_concurrency.py` |
| Fig. 14: KV-cache pressure, cancelled-output reuse, runtime overhead | E6 | `scripts/run_engine.py` |

Names: the paper's τ², Workspace and Micro. workloads are the suites `tau2long`, `workspace` and `micro`; WorkBench, Micro-short
and τ²-short are `workbench`, `tpcc` and `tau2`; Undo is the protocol `nexus_undo`; the serializable completion rate (SCR) is the
fraction of conflicting pairs that complete correctly within the task budget (`pair_correct / pairs` in the result files).

## Setup

```bash
python -m nexus.engine.sglang_plugin.launch --model-path <model> --tp 2 --port 30001 --context-length 65536 \
    --tool-call-parser <parser> --reasoning-parser <parser>
```

The paper's experiments use MiniMax-M2.7 and DeepSeek-V4-Flash, each served with two-way tensor parallelism and a 64k context
window. Prefix caching (RadixAttention) is on by default; `--max-total-tokens` sets the KV pool for the KV-pressure experiment.

Defaults of the drivers, which are the settings of the paper unless an experiment says otherwise: key window m = 8 (`KEY_WINDOW`)
and verification chunk b = 64 (`VERIFY_CHUNK`); sampling temperature 1.0 and top-p 0.95 (`--temperature`, `--top-p`); three seeds;
a per-task budget of 30 model turns (`--max-steps`) and five restarts after aborts (`Protocol.RETRY_CAP`; `atomix` keeps the
retry budget of its original runtime, three); a run that makes no progress for ten minutes is stopped (`STALL_S`). Exhausting a
budget fails the task and its pair.

## E1. Concurrency sweep (Fig. 8)

N agents in a closed loop, every task a member of a conflicting pair, three seeds, two tasks per agent (at least 16 tasks per run):

```bash
for B in tau2long workspace micro; do
  python scripts/run_concurrency.py $E --benchmark $B --protocols serial,twopl,occ,nexus_undo,mtpo,nexus --agents 2,4,8,16 --seeds 3
done
python scripts/make_figures.py          # results/figures/fig_concurrency_<B>.pdf: SCR, GPU time/task, tokens/task, JCT vs N
```

`serial` runs once at N=2 as the cost reference. The printed table and `summary.md` give per protocol and N: SCR, GPU seconds per
task, tokens per task, mean and p95 job completion time, invalidations and replays per run. The error bars of the figures are
half the range (maximum minus minimum) over the seeds, around the mean.

## E2. Anchored replay study (Fig. 9, Fig. 10)

Repair mechanisms on the three workloads at N=8 (replay = `nexus`, regeneration without drafts = `nexus_nodraft`, Undo =
`nexus_undo`, restart = `nexus_restart`), then the sensitivity of replay to the key window, the verification chunk and the
sampling temperature (content keys vs position keys) on τ²:

```bash
python scripts/run_replay_study.py $E --benchmarks tau2long,workspace,micro --seeds 3 \
    --repairs nexus,nexus_nodraft,nexus_undo,nexus_restart
python scripts/run_replay_study.py $E --benchmarks tau2long --seeds 3 --repairs "" \
    --windows 1,2,4,8,16,32 --chunks 8,16,32,64,128,256 --temperatures 0,0.25,0.5,0.75,1.0
```

The window sweep keeps b = 64 and the chunk sweep keeps m = 8. The summary (`results/replay_study_summary.jsonl` and the printed
table) reports per configuration: replays per run, draft acceptance (accepted / verified draft tokens), the share of the
invalidated suffix that was decoded again, and the outcome of each replay (same tokens, same tool calls, or a tool call changed).
Per-repair GPU time against suffix length comes from the `repair` events in the transcripts (`runs.jsonl`, task `calls`).

## E3. Transaction length (Fig. 11)

L function-disjoint Workspace tasks chained into one transaction, six task pairs per run, 64 model turns per task:

```bash
python scripts/run_concurrency.py $E --benchmark workspace --protocols serial,twopl,occ,nexus_undo,mtpo,nexus --agents 2,4,8 \
    --long 1,2,3,4 --max-steps 64 --seeds 3 --tasks 12 --tasks-per-agent 0
python scripts/make_figures.py          # results/figures/fig_length_workspace.pdf: tokens and GPU time relative to serial vs L
```

## E4. Short-conflict suites (Table II)

Two agents, one conflict per cell, every protocol, on 48 WorkBench cells and the first 10 instances of each of the five TPC-C and
five τ² conflict templates (50 cells each); every cell runs twice:

```bash
python scripts/run_short.py $E --cells workbench,tpcc,tau2 --per-template 10 \
    --protocols serial,naive,twopl,occ,mtpo,sbus,atomix,cordon,memtx,nexus --trials 2 --parallel 4
```

`results/short_<model>/summary.md` has correctness per cell and per protocol, speedup and token cost relative to serial, aborts,
deadlocks, notifications; leaked irreversible actions (the Leak columns) are the `leaked_unrecoverable` statistic. Without
`--per-template` every instance runs (48 + 90 + 200 cells).

## E5. Adaptive waiting (Fig. 13)

The waiting policy on τ²: cost-aware (`nexus`), always wait (`nexus_wait`), always proceed (`nexus_proceed`):

```bash
python scripts/run_concurrency.py $E --benchmark tau2long --agents 2,4,8,16 --seeds 3 --protocols nexus,nexus_wait,nexus_proceed
```

The cost breakdown at N=8 is read from the E1 runs on `tau2long`: the per-task `calls` records in `runs.jsonl` carry
input/cached/output tokens, draft accounting and preemption flags; `intent_wait_s` and `blocked_s` give waiting time; `repair_s`
in the transcript's `repair` events gives repair time, which together give decoding, waiting and repairing per protocol.

Further ablations of Nexus (repair mechanism, intents, engine priority), not reported in the paper, run the same way:

```bash
python scripts/run_concurrency.py $E --benchmark workspace --agents 2,4,8,16 --seeds 3 \
    --protocols nexus,nexus_restart,nexus_notify,nexus_undo,nexus_nodraft,nexus_nointents,nexus_nopriority
```

## E6. Engine effects and overheads (Fig. 14)

```bash
# KV pressure (Workspace, N=8): one server per pool size (--max-total-tokens 102400 / 65536 / 32768), then
python scripts/run_engine.py kv $E --servers 102k=http://H:30001/v1,64k=http://H:30002/v1,32k=http://H:30003/v1 --benchmark workspace
# reuse of cancelled generations' output as draft
python scripts/run_engine.py reuse --results 'results/conc_*'
# wall time of the Nexus layer per tool call and per commit (no model)
python scripts/run_engine.py overhead --agents 2,4,8,16
python scripts/make_figures.py          # results/figures/fig_kv_workspace.pdf
```

## E7. Second model (Fig. 12, Table II)

The same commands with another served model (`--model`, `--tokenizer`, and `--tool-format hermes` for Hermes-style tool calls):
E1 on `workspace`, and E4 with `--trials 1`.

## E8. Workload characterization (Fig. 3, Table I)

`scripts/profile_workloads.py` computes task-level statistics (duration, inference share, number of calls, tokens, read/write-set
sizes and the positions of reads and writes within a task) from trajectory files under `results/profiles/raw/<benchmark>/`. It has
loaders for the serial trajectories written by `run_short.py` / `run_concurrency.py` and for public agent traces downloaded
separately (TraceLab, Terminal-Bench, the official WorkBench trajectories and SWE-agent; the sources are listed in the script). The TraceLab traces are fetched with `bash scripts/fetch_third_party.sh traces`, and the
script needs duckdb (`pip install -e '.[profiling]'`).
