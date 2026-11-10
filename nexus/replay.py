"""Anchored replay of an invalidated task, on the trajectory the agent loop keeps.

The trajectory is the message list the model sees, annotated by the loop: a tool result carries the objects it observed (the
dependency record, `reads_of[index]`) and the store op it produced (`_op_seq`); an assistant turn carries its output token
ids when the engine exposes them (`_tokens`). Given the invalidated objects, replay

  1. rolls back: finds the earliest tool result s that observed an invalidated object, saves the suffix after it (including a
     cancelled generation's tokens) as the draft D, and retracts the buffered writes and intents issued in D (lines 1-3);
  2. re-reads: re-executes the call at s on the current snapshot and splices the updated result in (line 4);
  3. hands the agent loop the draft and the recorded tool calls of D: the engine verifies D against the updated context and
     re-decodes only where the replay diverges (lines 5-11); a tool call the model re-emits reuses its recorded result when its
     arguments and all of its versioned dependencies are unchanged, otherwise it executes again through the CC layer (lines 12-18).

The repair ablations share this module: `regen` is replay without a draft (every token is decoded again; memoization of
unchanged tool results stays), `undo` refreshes the stale results in place and retracts only the dependent writes (the model is told
which steps to redo), `notify` appends the invalidation and leaves the context as it is (model-driven reconciliation), `restart`
clears the context. All of them use the same dependency records and the same retraction of buffered writes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .engine.base import Draft
from .tools import conflicts, is_content_arg, wire_to_tool

PREEMPT_MSG = ("[NOTIFICATION] Your work was interrupted by the concurrency controller at this point: data you had read here "
               "was changed by an agent ordered before you, so your context was rolled back to this read (it has been "
               "refreshed) and any actions you took after it were undone. Details:\n{notes}\n[/NOTIFICATION]\n"
               "Continue the task from here using the updated data.")


@dataclass
class RecordedCall:
    """A tool call of the invalidated suffix: reused by memoization if its dependencies are unchanged at replay time."""
    tool: str
    args: dict
    result: str
    read_objs: set
    kind: str                       # read | write
    op_seq: int | None = None


@dataclass
class ReplayState:
    """What the agent loop carries while a replay is in progress."""
    draft: Draft | None
    calls: list = field(default_factory=list)      # RecordedCall, in trajectory order
    invalidated: set = field(default_factory=set)  # objects whose change triggered the replay
    replay_point: int = 0                          # index of the refreshed tool result
    memo_hits: int = 0
    reexecuted: int = 0
    started_at_commit_ts: int = 0

    def memoized(self, tool: str, args: dict, unchanged) -> RecordedCall | None:
        """The recorded result for (tool, args) if the recorded read's dependencies are all unchanged (`unchanged(objs)`)."""
        key = json.dumps(args, sort_keys=True, default=str)
        for c in self.calls:
            if c.kind != "read" or c.tool != tool or json.dumps(c.args, sort_keys=True, default=str) != key:
                continue
            if any(conflicts(o, x) for o in c.read_objs for x in self.invalidated):
                return None
            if not unchanged(c.read_objs):
                return None
            self.memo_hits += 1
            return c
        return None


def affected_indices(reads_of: dict, objs: set) -> list:
    return sorted(i for i, r in reads_of.items() if any(conflicts(o, x) for o in objs for x in r))


def seqs_after(messages: list, idx: int) -> list:
    """Store ops of the write results after message `idx` (exact, by op sequence)."""
    return [m["_op_seq"] for j, m in enumerate(messages) if j > idx and m.get("role") == "tool" and m.get("_op_seq") is not None]


def _call_of(messages: list, tool_call_id: str):
    for j, m in enumerate(messages):
        if m["role"] == "assistant":
            for tc in m.get("tool_calls", []):
                if tc["id"] == tool_call_id:
                    return j, tc
    return None, None


def _suffix_tokens(dropped: list, preempted_tokens: list | None) -> list:
    toks = []
    for m in dropped:
        if m.get("role") == "assistant" and m.get("_tokens"):
            toks += list(m["_tokens"])
    if preempted_tokens:
        toks += list(preempted_tokens)
    return toks


def _recorded_calls(messages: list, start: int, specs) -> list:
    out = []
    names = {tc["id"]: tc for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls", [])}
    for j in range(start, len(messages)):
        m = messages[j]
        if m.get("role") != "tool":
            continue
        tc = names.get(m.get("tool_call_id"))
        if not tc:
            continue
        tool = wire_to_tool(tc["name"])
        spec = specs.get(tool)
        kind = spec.kind if spec is not None else "write"
        out.append(RecordedCall(tool, tc.get("args", {}), m.get("content", "").split("\n\n[NOTIFICATION]")[0], set(m.get("_reads", ())), kind, m.get("_op_seq")))
    return out


def note_block(notes) -> str:
    return PREEMPT_MSG.format(notes="\n".join(f"- {n}" for n in notes) if notes else "- (see updated data below)")


def rollback_and_reread(protocol, name, messages, reads_of, notes, objs, task, run, preempted_tokens=None, drafts=True, chunk=64):
    """Roll back to the earliest stale read and re-read it. Returns (messages, reads_of, dropped_count, ReplayState | None).
    With `drafts=False` the suffix is still memoized but no draft is handed to the engine (regeneration without drafts)."""
    specs = protocol.store.backend.specs
    aff = affected_indices(reads_of, objs)
    if not aff:
        messages.append({"role": "user", "content": note_block(notes)})
        return messages, reads_of, 0, None
    i = aff[0]
    tc_id = messages[i]["tool_call_id"]
    j, call = _call_of(messages[:i], tc_id)
    tool = wire_to_tool(call["name"]) if call else ""
    if call and specs.get(tool) is not None and specs[tool].kind == "write":
        # the replay point is a WRITE whose own footprint read was invalidated: it was computed on stale data. Retract it and the
        # suffix, and cut the context back to before the turn that issued it; the draft starts at that turn.
        dropped = messages[j:]
        protocol.retract_seqs(name, seqs_after(messages, j - 1))
        kept = messages[:j] + [{"role": "user", "content": note_block(notes)}]
        st = ReplayState(Draft(_suffix_tokens(dropped, preempted_tokens), 0, chunk) if drafts else None,
                         _recorded_calls(messages, j, specs), set(objs), j)
        return kept, {k: v for k, v in reads_of.items() if k < j}, len(messages) - j, _finish(st)
    dropped = messages[i + 1:]
    protocol.retract_seqs(name, seqs_after(messages, i))
    fresh = None
    if call:
        try:
            fresh = protocol.call(name, tool, call["args"])
            requeue_uncovered(protocol, name, fresh, objs, run)
        except Exception:
            fresh = None
    kept = messages[:i]
    content = (fresh.result if fresh else messages[i]["content"].split("\n\n[NOTIFICATION]")[0]) + "\n\n" + note_block(notes)
    kept.append({"role": "tool", "tool_call_id": tc_id, "content": content, "_reads": set(fresh.read_objs) if fresh else set(reads_of.get(i, ()))})
    if run is not None:
        run.transcript.append({"role": "tool", "tool": tool, "args": call["args"] if call else {},
                               "result": (fresh.result if fresh else "<re-read failed: stale result kept>")[:500],
                               "notes": list(fresh.notifications) if fresh else [], "framework": "replay_reread"})
    new_reads = {k: v for k, v in reads_of.items() if k < i}
    if fresh:
        new_reads[i] = set(fresh.read_objs)
    st = ReplayState(Draft(_suffix_tokens(dropped, preempted_tokens), 0, chunk) if drafts else None,
                     _recorded_calls(messages, i + 1, specs), set(objs), i)
    return kept, new_reads, len(messages) - len(kept), _finish(st)


def _finish(st: ReplayState) -> ReplayState | None:
    if st.draft is not None and not st.draft.tokens:
        st.draft = None            # nothing to verify (no token-level engine, or the suffix was empty): plain regeneration
    return st


def restart(protocol, name, messages, task, notes):
    """The `restart` ablation: undo everything this attempt wrote and clear the context."""
    seqs = set(seqs_after(messages, -1))
    if getattr(protocol, "SHADOW_WRITES", False):
        seqs |= {o.seq for o in protocol.store.ops_of(name)}
    protocol.retract_seqs(name, seqs)
    return [{"role": "user", "content": task + "\n\n" + note_block(notes)}], {}, len(messages) - 1


def undo_in_place(protocol, name, messages, reads_of, notes, objs, run):
    """The `undo` ablation (selective undo): refresh each stale tool result in place, retract only the buffered writes that touch the
    invalidated objects and mark their results as retracted, and ask the model to redo just those; independent later steps stay."""
    specs = protocol.store.backend.specs
    aff = affected_indices(reads_of, objs)
    if not aff:
        messages.append({"role": "user", "content": note_block(notes)})
        return messages, reads_of, 0
    retracted = protocol.retract_objs(name, set(objs))
    ret_keys = {(o.tool, json.dumps(o.kwargs, sort_keys=True, default=str)) for o in retracted}
    calls = {tc["id"]: tc for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls", [])}
    new_reads = dict(reads_of)
    changed = []
    for i, m in enumerate(messages):
        if m["role"] != "tool":
            continue
        tc = calls.get(m["tool_call_id"])
        if not tc:
            continue
        tool = wire_to_tool(tc["name"])
        key = (tool, json.dumps(tc["args"], sort_keys=True, default=str))
        if i in aff and specs.get(tool) is not None and specs[tool].kind == "read":
            try:
                fresh = protocol.call(name, tool, tc["args"])
                requeue_uncovered(protocol, name, fresh, objs, run)
                m["content"] = "[REFRESHED after a concurrent change; this is the current value]\n" + fresh.result
                m["_reads"] = set(fresh.read_objs)
                if run is not None:
                    run.transcript.append({"role": "tool", "tool": tool, "args": tc["args"], "result": fresh.result[:500], "notes": list(fresh.notifications), "framework": "undo_refresh", "step": i})
                new_reads[i] = set(fresh.read_objs); changed.append(f"step {i}: re-read {tool}")
            except Exception:
                pass
        elif key in ret_keys:
            m["content"] = ("[RETRACTED by the concurrency controller: this action was based on data that another agent "
                            "changed. Its effect has been undone; redo it against the refreshed data if it is still needed]")
            tc["args"] = redo_args(tc["args"])
            changed.append(f"step {i}: retracted {tool}")
    summary = "\n".join(f"- {c}" for c in changed) or "- (nothing to redo)"
    messages.append({"role": "user", "content": note_block(notes) + "\n\nWhat the controller changed in your history:\n" + summary +
                     "\nSteps not listed above are still valid and must NOT be repeated."})
    return messages, new_reads, len(retracted)


def redo_args(a: dict) -> dict:
    """Identifying arguments of a retracted write; content-bearing arguments are withheld (a model handed the old content copies it
    back verbatim over the refreshed object)."""
    out = {}
    for k, v in (a or {}).items():
        if is_content_arg(k, v):
            out[k] = "<re-derive from the refreshed data>"
        else:
            out[k] = v
    return out


def requeue_uncovered(protocol, name, res, objs, run=None):
    """Notifications drained by a framework-issued call inside a repair must reach the agent through its next real tool return,
    unless every object they concern is one this repair already handles (delivering it again would only trigger the same repair)."""
    notes = getattr(res, "notifications", None)
    if not notes:
        return
    note_objs = getattr(res, "note_objs", None) or set()
    if note_objs and objs and all(any(conflicts(o, x) for x in objs) for o in note_objs):
        if run is not None:
            run.notes_absorbed = getattr(run, "notes_absorbed", 0) + len(notes)
        return
    protocol.requeue(name, notes, note_objs)
