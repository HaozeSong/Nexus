# Design

This document explains how the components of `nexus/` fit together. It is self-contained; the module docstrings carry the
details of each piece.

## 1. Execution model

A task is one agent working through one request with tool calls. Under Nexus every task is a transaction:

* **Snapshot.** `store.py` is a multi-version log of operations. A task reads the committed state overlaid with its own buffered
  (private) operations; other tasks' buffered operations are invisible to it (`materialize(viewer=task)`).
* **Buffered writes.** A tool call that writes executes immediately against the task's private view and is recorded as a buffered
  op. Commit publishes the task's ops in commit order (`manager.py`, `commit`), so the live state is the replay of committed ops.
* **Access sets.** Every tool declares what it reads and writes in terms of records (`table:id`), listings (`table:*`) and inserts
  (`table:+`), resolved against the task's snapshot before execution (`tools.py`). Listings conflict with inserts and with
  updates of listed rows; records conflict with their own updates.
* **Correctness.** A set of concurrent tasks is correct when the final state equals that of some serial execution of the same
  tasks (`serial.py` evaluates the two orders of a pair through reference procedures).

## 2. Versioned read sets

`readsets.py` keeps, per task, every record the task has observed and the version it saw: committed records carry the commit
timestamp of their last writer, the task's own buffered writes carry a pin that is resolved at commit, and listings carry the
version of the table. Four rules keep the set sound:

1. a read through the task's own buffered write records the buffered version, not the committed one;
2. a record the task wrote and then read is validated against the committed version at commit, not against its own write;
3. before a write executes, the records it depends on are checked for staleness (**premise check**): a write on a stale premise is
   refused with "Action NOT executed" and the task is repaired first;
4. when a stale read is detected, the buffered writes that depend on it are retracted (`manager.py`, `_retract_exact`) so that a
   replayed task cannot commit a write derived from a value it no longer believes.

## 3. Forward validation and cancellation

When a task commits, the manager intersects the committed write set with the read set of every running task
(`manager.py`, `commit`, `_invalidate`). A task whose read set is hit is marked stale immediately rather than at its own commit: its
in-flight generation is cancelled through the engine (`engine/*.abort`), the partial output of the cancelled request is kept, and
the task is repaired before it issues another tool call. The reads an in-flight request depends on are the reads in its prompt
(`begin_generation` records them as `ctx.gen_reads`). Validation at commit then only has to catch what forward validation already
announced, so a task that reached commit without an invalidation commits.

## 4. Anchored replay

`replay.py` and `agent.py` implement the repair. The agent loop annotates its message list: each tool result carries the records it
observed (`_reads`) and the buffered op it produced (`_op_seq`); each assistant turn carries its output token ids (`_tokens`) when
the engine exposes them. On an invalidation:

1. find the earliest tool result whose reads are stale (`affected_indices`);
2. drop everything after it, retract the buffered writes and intents in the dropped suffix, and keep the suffix's tokens (plus
   the cancelled generation's partial output) as a **draft**;
3. re-execute the stale read against the refreshed snapshot;
4. continue generating with the draft: the engine verifies up to `VERIFY_CHUNK` draft tokens per forward pass and accepts the
   prefix the keyed sampler would have produced, substitutes the first diverging token, and re-aligns the draft (`sampling.align`:
   the old continuation after the substitution if it matches, otherwise the longest match of the last `KEY_WINDOW` tokens);
5. a tool call the model re-issues with the same arguments whose dependencies have not changed since the recorded call is served
   from the record (`ReplayState.memoized`, `NexusTM.memo_read`) instead of executed again.

The other repair mechanisms share the same annotations and differ only in step 2 and 4: `regen` drops the draft; `undo` refreshes
stale results in place and asks the model to redo the dependent steps; `notify` appends the invalidation to the context; `restart`
clears it. They are selected by `NexusTM.REPAIR` and exposed as the `nexus_*` protocols (`protocols/nexus/ablations.py`).

## 5. Content-keyed sampling

A draft is only useful if unchanged reasoning reproduces token-for-token under sampling. `sampling.py` separates the model's
prediction from the randomness: the Gumbel noise added to each candidate's log-probability is a pseudo-random function of
`(seed, key, token)`, where the **content key** is a hash of the last *m* output tokens and the number of earlier occurrences of
that window in the sequence (so repeated windows get fresh noise). Two contexts that agree locally therefore draw the same noise
regardless of absolute position, which is what a replay after a longer or shorter tool result needs. Verification applies the
same rule to the model's distribution at each draft position: the draft token is accepted iff it is the arg-max of
log-probability plus noise, so accepted tokens are exactly the tokens ordinary keyed decoding would have produced. Position keys
(`position_keys=True`) are the deterministic mode of common engines and are kept for the ablation. The engine-side half
(`engine/sglang_plugin/`) draws the same noise inside the server so that ordinary decoding and verification agree.

## 6. Write intents

`intents.py` tracks pending writes before their values exist: a task that will write a record registers an **intent** when it
first reads the record (or when its tool schema declares the write). A younger task that reads a record an older task holds an
intent on may **wait** for the older task's commit instead of reading a value that is about to change; waits are ordered by task
age (younger waits for older only), so the wait-for graph is acyclic. Registration also invalidates younger tasks that have
already read the record, so they are repaired early. Whether to wait is a priced decision: the reader proceeds when the expected
cost of a later repair (probability that the record changes times the mean replay cost observed for this intent table) is below
the cost of waiting (expected remaining hold time of the intent, times a factor derived from engine load that is zero when the
engine has queued requests, because waiting then costs no GPU time). Policies: `priced` (default), `wait`, `proceed`. Tasks that
hold intents other tasks wait on get engine scheduling priority (`NexusTM.priority`).

## 7. Liveness and irreversible effects

A task whose commit fails `MAX_REPAIRS` times aborts (counted as a failure). Tools with irreversible external effects
(`unrecoverable` in the tool spec) validate the whole read set before execution, so an irreversible action is never taken on a
stale premise; committing them after validation is the backend's responsibility.

## 8. Engines and the agent loop

`engine/base.py` is the interface: `chat(system, messages, tools, rid, cancel_event, draft, sampling, priority)` returns a `Turn`
with text, tool calls, usage, token ids and draft accounting; `abort(rid)` cancels. `engine/sglang_native.py` implements it over
SGLang's native `/generate` endpoint (prompt built with the model's chat template, raw-output tool-call parsing, logprob-based
verification of draft chunks, keyed client-side sampling, abort). `engine/openai_chat.py` covers OpenAI-compatible servers without
token ids. `engine/sim.py` is a simulated token-level engine around a scripted policy (used by the tests), and `engine/scripted.py`
drives deterministic no-model agents. `engine/telemetry.py` samples the server's queue depth and KV usage for GPU-time attribution
and the intent pricing.

## 9. Workload driver

`workload.py` runs N agents in a closed loop over a task pool: a fraction `p_c` of the tasks are members of conflicting pairs,
the second member of a pair is released at a configurable point of the first (`t0`, `after_read`, `mid`, `pre_commit`, or an
empirical distribution), and L function-disjoint tasks can be chained into one transaction. It records per-task latency, tokens,
tool calls with draft accounting, protocol statistics and the pair verdicts.
