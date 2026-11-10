"""The agent loop: turns of LLM inference and tool calls over a trajectory, wired to a transaction manager and an engine.

Every tool call passes through the manager (`protocol.call`); every generation is registered with it (`begin_generation`) so that
forward validation can cancel it; a conflict is repaired on the trajectory according to the manager's REPAIR mode:

  replay   anchored replay: roll back to the earliest stale read, keep the suffix as a draft, re-read, and continue with the
           engine verifying the draft; recorded tool results are reused when their dependencies are unchanged
  regen    the same rollback and memoization, but no draft (every token is decoded again)
  undo     selective undo: refresh stale results in place, retract only dependent writes, ask the model to redo them
  notify   append the invalidation to the context and let the model reconcile (MTPO-style)
  restart  clear the context and start the task over (classic abort/retry)

Baselines that abort (2PL, OCC, S-Bus, Atomix) raise AbortAgent from the manager; the loop clears the context and retries up to
the manager's retry cap.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

from .engine.base import SamplingSpec, Usage
from .replay import ReplayState, note_block, restart, rollback_and_reread, undo_in_place
from .tools import conflicts, wire_to_tool
from .transaction import AbortAgent, AgentFailed, Protocol

SYSTEM = """You are an office-automation assistant operating on the company's shared systems \
(calendar, email, CRM, project management, analytics) through tools. The current date and time is \
2023-11-30 00:00:00. Dates use the format YYYY-MM-DD and datetimes YYYY-MM-DD HH:MM:SS.

Complete the user's task fully and precisely: look up what you need, then make the changes. \
Act immediately; never ask for confirmation. When the task is complete, reply with a one-sentence \
summary and make no further tool calls.

Other agents may be working on the same systems at the same time. If a tool result contains a \
[NOTIFICATION] block, another agent has changed data that you read earlier. Decide whether that \
change invalidates any premise of your plan or any action you already took: if it does, issue the \
corrective reads/writes needed so that the task outcome is still correct given the new data; if it \
does not affect your task, simply continue. If a result says your previous attempt was aborted, all \
of its effects were rolled back and you must redo the task from the beginning."""

ABORT_MSG = ("[SYSTEM] Your previous attempt was aborted by the concurrency controller because of a "
             "conflict with another agent; all of its effects were rolled back. Redo the task from scratch.")


@dataclass
class AgentRun:
    name: str
    task: str
    status: str = "running"          # ok | failed | max_steps | error
    final_text: str = ""
    usage: Usage = field(default_factory=Usage)
    tool_calls: int = 0
    aborts: int = 0
    arg_errors: int = 0
    blind_writes: int = 0
    notifications: int = 0
    blocked_s: float = 0.0
    wall_s: float = 0.0
    attempts: int = 0
    transcript: list = field(default_factory=list)
    error: str = ""
    wasted: Usage = field(default_factory=Usage)     # tokens of attempts that were later aborted
    attempt_usage: Usage = field(default_factory=Usage)
    preemptions: int = 0
    preempted_output_tokens: int = 0
    rollback_dropped_msgs: int = 0
    nudges: int = 0
    calls: list = field(default_factory=list)        # per-request telemetry
    # anchored replay accounting
    replays: int = 0
    invalidated_tokens: int = 0                       # tokens of invalidated suffixes (drafts offered)
    draft_accepted: int = 0
    draft_verified: int = 0
    draft_decoded: int = 0
    memo_hits: int = 0
    reexecuted_calls: int = 0
    replay_outcomes: list = field(default_factory=list)   # per replay: (draft_tokens, accepted, decoded, tool_calls_changed)


def _bad_args(params: dict | None, args: dict) -> str:
    if not params or not isinstance(args, dict):
        return ""
    props = params.get("properties") or {}
    unknown = [k for k in args if k not in props]
    missing = [k for k in (params.get("required") or []) if k not in args]
    if unknown or missing:
        parts = []
        if unknown:
            parts.append(f"unexpected argument(s) {unknown}")
        if missing:
            parts.append(f"missing required argument(s) {missing}")
        return "; ".join(parts) + f". Valid parameters: {sorted(props)}"
    return ""


def repair_mode(protocol) -> str:
    """The manager's repair mode (Nexus: REPAIR; other protocols: their ROLLBACK setting)."""
    m = getattr(protocol, "REPAIR", None)
    if m:
        return m
    return {"append": "notify", "truncate": "regen", "restart": "restart", "surgical": "undo", "auto": "regen"}.get(getattr(protocol, "ROLLBACK", "append"), "notify")


def _seed(trial_id: str, name: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{trial_id}:{name}".encode(), digest_size=8).digest(), "little") & 0x7FFFFFFFFFFFFFFF


def run_agent(protocol: Protocol, name: str, task: str, llm, max_steps: int = 30, tool_names: list[str] | None = None,
              system_extra: str = "", trial_id: str = "t") -> AgentRun:
    run = AgentRun(name=name, task=task)
    style = "anthropic" if getattr(llm, "backend", "") == "anthropic" else "openai"
    tools = protocol.store.backend.tool_schemas(style, tool_names)
    schema_params = {t.get("name") or t["function"]["name"]: (t.get("input_schema") or t.get("function", {}).get("parameters") or {}) for t in tools}
    if getattr(protocol, "RETRACT_TOOL", False):
        from .tools import schemas_from
        n, d, p = protocol.RETRACT_SCHEMA
        tools += schemas_from({n: (d, p)}, style)
    system = SYSTEM + ("\n\n" + system_extra if system_extra else "")
    mode = repair_mode(protocol)
    sampling = SamplingSpec(seed=_seed(trial_id, name), window=getattr(protocol, "KEY_WINDOW", 8),
                            position_keys=getattr(protocol, "POSITION_KEYS", False))
    chunk = getattr(protocol, "VERIFY_CHUNK", 64)
    drafts = bool(getattr(protocol, "DRAFTS", True)) and mode == "replay" and getattr(llm, "supports_tokens", False)
    specs = protocol.store.backend.specs
    t0 = time.monotonic()
    messages = [{"role": "user", "content": task}]
    reads_of: dict[int, set] = {}        # index of a tool result -> objects it observed (dependency records)
    replay: ReplayState | None = None
    steps = 0
    ncall = 0

    def rollback(notes, objs, preempted_tokens=None):
        """Repair the trajectory after an invalidation of `objs`."""
        nonlocal messages, reads_of, replay
        t_rep = time.monotonic()
        if mode == "restart" or not objs:
            if mode == "restart":
                messages, reads_of, dropped = restart(protocol, name, messages, task, notes)
            else:
                messages.append({"role": "user", "content": note_block(notes)}); dropped = 0
            replay = None
        elif mode == "undo":
            messages, reads_of, dropped = undo_in_place(protocol, name, messages, reads_of, notes, objs, run)
            replay = None
        elif mode in ("replay", "regen"):
            messages, reads_of, dropped, st = rollback_and_reread(protocol, name, messages, reads_of, notes, objs, task, run,
                                                                 preempted_tokens=preempted_tokens, drafts=drafts, chunk=chunk)
            if st is not None:
                st.started_at_commit_ts = getattr(protocol, "commit_ts", 0)
                run.replays += 1
                if st.draft is not None:
                    run.invalidated_tokens += len(st.draft.tokens)
                else:
                    run.invalidated_tokens += sum(len(m.get("_tokens") or ()) for m in [])
            replay = st
        else:   # notify
            messages.append({"role": "user", "content": note_block(notes)})
            dropped = 0
            replay = None
        run.rollback_dropped_msgs += dropped
        run.transcript.append({"role": "system", "event": "repair", "mode": mode, "dropped": dropped,
                               "draft_tokens": (len(replay.draft.tokens) if replay and replay.draft else 0),
                               "repair_s": round(time.monotonic() - t_rep, 3)})

    try:
        while True:
            run.attempts += 1
            restarted = False
            while steps < max_steps:
                steps += 1
                ncall += 1
                rid = f"{trial_id}-{name}-{ncall}-{uuid.uuid4().hex[:8]}"
                cancel = threading.Event()
                prompt_reads = set().union(*reads_of.values()) if reads_of else set()
                protocol.begin_generation(name, rid, cancel, prompt_reads)
                draft = replay.draft if (replay is not None and replay.draft is not None and replay.draft.remaining() > 0) else None
                try:
                    turn = llm.chat(system, messages, tools, rid=rid, cancel_event=cancel, draft=draft, sampling=sampling,
                                    priority=(protocol.priority(name) if hasattr(protocol, "priority") else 0))
                finally:
                    protocol.end_generation(name)
                run.usage.add(turn.usage)
                run.attempt_usage.add(turn.usage)
                run.calls.append({"rid": rid, "t_start": turn.t_start, "t_end": turn.t_end, "ttft_s": round(turn.ttft_s, 3),
                                  "gen_s": round(turn.gen_s, 3), "input": turn.usage.input_tokens, "cached": turn.usage.cache_read,
                                  "output": turn.usage.output_tokens, "preempted": turn.preempted,
                                  "partial_output": turn.partial_output_tokens, "n_msgs": len(messages),
                                  "draft_accepted": turn.draft_accepted, "draft_verified": turn.draft_verified, "draft_decoded": turn.draft_decoded})
                if draft is not None:
                    run.draft_accepted += turn.draft_accepted; run.draft_verified += turn.draft_verified; run.draft_decoded += turn.draft_decoded
                    draft.accepted += turn.draft_accepted; draft.verified += turn.draft_verified; draft.decoded += turn.draft_decoded
                    draft.cursor = turn.draft_cursor
                if turn.preempted:
                    run.preemptions += 1
                    run.preempted_output_tokens += turn.partial_output_tokens
                    notes, objs = protocol.drain(name)
                    run.notifications += len(notes)
                    run.transcript.append({"role": "system", "event": "preempted", "rid": rid, "partial": turn.partial_output_tokens,
                                           "notes": notes, "repair": mode})
                    rollback(notes, objs, preempted_tokens=turn.token_ids)
                    continue
                amsg = {"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls,
                        "_out": turn.usage.output_tokens, "_in": turn.usage.input_tokens, "_cached": turn.usage.cache_read}
                if turn.token_ids is not None:
                    amsg["_tokens"] = list(turn.token_ids)
                if getattr(turn, "reasoning", None):
                    amsg["reasoning_content"] = turn.reasoning
                messages.append(amsg)
                run.transcript.append({"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls,
                                       "draft": ({"accepted": turn.draft_accepted, "verified": turn.draft_verified, "decoded": turn.draft_decoded} if draft is not None else None)})
                if not turn.tool_calls and run.nudges < 2 and (
                        run.tool_calls == 0 or re.search(r"\b(let me (?:start|begin|first|now)|I'?ll (?:start|begin|now|first)|I will (?:start|begin|now|first)|starting with|first,? (?:let me|I))\b", turn.text or "", re.I)):
                    run.nudges += 1
                    messages.append({"role": "user", "content": "You have not completed the request(s) with the tools yet. Continue now: "
                                     "call the tools needed to carry out every request, then report."})
                    run.transcript.append({"role": "system", "event": "nudge"})
                    continue
                if not turn.tool_calls:
                    try:
                        notes = protocol.commit(name)
                    except AbortAgent as e:
                        protocol.abort(name, e.reason)
                        run.aborts += 1
                        run.wasted.add(run.attempt_usage); run.attempt_usage = Usage()
                        messages = [{"role": "user", "content": task + "\n\n" + ABORT_MSG + f" (reason: {e.reason})"}]
                        reads_of = {}; replay = None
                        run.transcript.append({"role": "system", "event": "abort", "reason": e.reason})
                        restarted = True
                        break
                    if notes:
                        run.notifications += len(notes)
                        run.transcript.append({"role": "notification", "notes": notes})
                        objs = (protocol.last_note_objs or {}).get(name, set()) if mode in ("replay", "regen", "undo", "restart") else set()
                        if objs:
                            rollback(notes, objs)
                            continue
                        messages.append({"role": "user", "content": _notif_block(notes) +
                                         "\nYou had reported the task complete; re-check it against this change and continue."})
                        continue
                    if replay is not None:
                        run.replay_outcomes.append((len(replay.draft.tokens) if replay.draft else 0, replay.draft.accepted if replay.draft else 0,
                                                    replay.draft.decoded if replay.draft else 0, replay.reexecuted))
                        run.memo_hits += replay.memo_hits; run.reexecuted_calls += replay.reexecuted
                    run.final_text = turn.text
                    run.status = "ok"
                    break
                for tc in turn.tool_calls:
                    tool = wire_to_tool(tc["name"])
                    run.tool_calls += 1
                    bad = _bad_args(schema_params.get(tc["name"]), tc["args"])
                    rbw = getattr(protocol.store.backend, "read_before_write", None)
                    if not bad and rbw and tool in rbw:
                        spec_w = specs.get(tool)
                        _, wobjs = spec_w.declared(tc["args"]) if spec_w else (set(), set())
                        seen = set().union(*reads_of.values()) if reads_of else set()
                        if wobjs and not any(conflicts(w, o) for w in wobjs for o in seen):
                            bad = (f"rewriting {', '.join(sorted(wobjs))} requires reading its CURRENT content first in this task "
                                   f"(other engineers may have changed it); call read_file and then rewrite it")
                            run.blind_writes += 1
                    res = None
                    memo = None
                    try:
                        if bad:
                            raise TypeError(bad)
                        if replay is not None and hasattr(protocol, "memo_read"):
                            memo = replay.memoized(tool, tc["args"], lambda objs: protocol.unchanged_since(objs, replay.started_at_commit_ts))
                        if memo is not None:
                            res = protocol.memo_read(name, tool, tc["args"], memo.result)
                        else:
                            res = protocol.call(name, tool, tc["args"])
                            if replay is not None:
                                replay.reexecuted += 1
                    except TypeError as e:
                        res_text, notes = f"Error: {e}", []
                        run.arg_errors += 1
                    except AbortAgent as e:
                        protocol.abort(name, e.reason)
                        run.aborts += 1
                        run.wasted.add(run.attempt_usage); run.attempt_usage = Usage()
                        messages = [{"role": "user", "content": task + "\n\n" + ABORT_MSG + f" (reason: {e.reason})"}]
                        reads_of = {}; replay = None
                        run.transcript.append({"role": "system", "event": "abort", "reason": e.reason})
                        restarted = True
                        break
                    except KeyError:
                        res_text, notes = f"Error: unknown tool {tool}", []
                    else:
                        res_text, notes = res.result, res.notifications
                        run.blocked_s += res.blocked_s
                    content = res_text
                    if notes:
                        run.notifications += len(notes)
                        content += "\n\n" + _notif_block(notes)
                    tmsg = {"role": "tool", "tool_call_id": tc["id"], "content": content}
                    if res is not None and getattr(res, "op_seq", None) is not None:
                        tmsg["_op_seq"] = res.op_seq
                    if res is not None and getattr(res, "read_objs", None):
                        tmsg["_reads"] = set(res.read_objs)
                        reads_of[len(messages)] = set(res.read_objs)
                    messages.append(tmsg)
                    run.transcript.append({"role": "tool", "tool": tool, "args": tc["args"], "result": res_text[:500], "notes": notes,
                                           **({"memoized": True} if memo is not None else {})})
                    if notes and mode in ("replay", "regen", "undo", "restart") and getattr(res, "note_objs", None):
                        nm_before = len(messages)
                        rollback(notes, res.note_objs)
                        if run.rollback_dropped_msgs or len(messages) != nm_before:
                            run.transcript.append({"role": "system", "event": "savepoint_rollback"})
                            break      # the remaining calls of this turn were derived from the stale context
                if restarted:
                    break
                if run.status == "ok":
                    break
            if run.status == "ok":
                break
            if restarted:
                continue
            run.status = "max_steps"
            break
    except AgentFailed as e:
        run.status, run.error = "failed", str(e)
    except Exception as e:
        run.status, run.error = "error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"
    finally:
        protocol.finish(name)
        run.wall_s = time.monotonic() - t0
    return run


def _notif_block(notes: list[str]) -> str:
    return "[NOTIFICATION]\n" + "\n".join(f"- {n}" for n in notes) + "\n[/NOTIFICATION]"
