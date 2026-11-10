"""Scripted-agent helpers shared by the per-benchmark policy modules.

Each role is driven by an *ideal self-healing* scripted policy: it reads, then (after a short pause so
both agents have read) writes; it reacts to [NOTIFICATION] blocks by re-reading and repairing exactly the
dependent write; after an abort its context is fresh and it starts over.  Expected behaviour:
serial, 2pl (with deadlocks), occ (with aborts), mtpo (with notifications) and sbus (with 409s) are correct; naive shows the anomaly.
"""
from __future__ import annotations

import json
import threading
import time

from .base import Engine, Turn, Usage

PAUSE = 0.4


def tool_history(messages):
    """[(tool, args, result_content)] in order, restarted after the agent's last retract-all (the scripted
    agent then redoes its task from scratch on the refreshed view)."""
    names = {}
    out = []
    for m in messages:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls", []):
                names[tc["id"]] = (tc["name"].replace("__", "."), tc["args"])
        elif m["role"] == "tool":
            n, a = names[m["tool_call_id"]]
            if n == "nexus.retract_my_actions":
                out = []
                continue
            out.append((n, a, m["content"]))
    return out


WRITE_HINTS = ("update", "delete", "create", "add_", "send", "place_order", "set_", "record_payment", "deliver", "cancel", "book", "reply", "forward", "retract")


def _my_writes_since_restart(messages):
    return sum(1 for n, _, _ in tool_history(messages) if any(k in n for k in WRITE_HINTS))


READ_TOOLS = ("search_", "get_")


def _last_read_index(messages):
    idx = -1
    names = {}
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            for tc in m.get("tool_calls", []):
                names[tc["id"]] = tc["name"]
        elif m["role"] == "tool" and any(k in names.get(m["tool_call_id"], "") for k in READ_TOOLS):
            idx = i
    return idx


_tls = threading.local()


def _count_notes(messages):
    return sum(1 for m in messages if isinstance(m.get("content"), str) and "[NOTIFICATION]" in m["content"])


def notified(messages):
    """True on the first policy call after a new notification arrived (the ScriptedLLM marks it handled
    once the policy has answered, so a repair is issued exactly once per notification)."""
    return getattr(_tls, "new_note", False)


def prefix_record(messages, table: str, rid: str | None = None):
    """The record 'as left by the agents ordered before you' from the latest notification (MTPO), as a dict."""
    # a refused action ("Action NOT executed: its premise changed") means the protocol requires a fresh READ before acting:
    # the value carried by the notification must not be used as a substitute for that read
    for m in reversed(messages):
        if m.get("role") == "tool":
            if str(m.get("content", "")).startswith("Action NOT executed"):
                return None
            break
    for line in reversed(note_text(messages).splitlines()):
        if line.startswith(f"{table} record ") and "as left by the agents ordered before you: " in line:
            if rid is None or f" record {rid} " in line:
                return json.loads(line.split("as left by the agents ordered before you: ", 1)[1])
    return None


def note_text(messages):
    notes = [m["content"] for m in messages if isinstance(m.get("content"), str) and "[NOTIFICATION]" in m["content"]]
    return notes[-1] if notes else ""


def clean(s: str) -> str:
    """Tool result without any notification block that rode on it."""
    return s.split("\n\n[NOTIFICATION]")[0].strip()


def parse_json(s):
    try:
        return json.loads(s.split("\n\n[NOTIFICATION]")[0])
    except Exception:
        return None


def shift(ts: str, minutes: int) -> str:
    import datetime as dt
    return (dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") + dt.timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")



class ScriptedLLM(Engine):
    backend = "scripted"
    supports_tokens = False
    gen_time = 0.0     # simulated decode time per call (seconds); cancellable in 20 ms slices

    def __init__(self, policy, gen_time: float = 0.0):
        self.policy = policy
        self.n = 0
        self.gen_time = gen_time

    def chat(self, system, messages, tools, rid=None, cancel_event=None, draft=None, sampling=None, priority=0) -> Turn:
        t0 = time.time()
        turn = self._chat(system, messages, tools)
        # simulate a long generation before a WRITE action (the unvalidated read->write window); cancellable
        if self.gen_time > 0 and any(any(k in c["name"] for k in WRITE_HINTS) for c in turn.tool_calls):
            slices = int(self.gen_time / 0.02)
            for i in range(slices):
                if cancel_event is not None and cancel_event.is_set():
                    self.n -= 0
                    return Turn("", [], Usage(0, i, 0, 0, 1, 0.0), None, "preempted", preempted=True,
                                partial_output_tokens=i, rid=rid, gen_s=time.time() - t0, t_start=t0, t_end=time.time())
                time.sleep(0.02)
        turn.rid = rid; turn.t_start = t0; turn.t_end = time.time(); turn.gen_s = turn.t_end - t0
        return turn

    def _chat(self, system, messages, tools) -> Turn:
        n = _count_notes(messages)
        if n < getattr(self, "seen_notes", 0):
            self.seen_notes = 0            # context was cleared by an abort
        _tls.new_note = n > getattr(self, "seen_notes", 0)
        self.n += 1
        has_retract = any(("retract_my_actions" in (t.get("name") or t.get("function", {}).get("name", ""))) for t in tools)
        if _tls.new_note and "interrupted by the concurrency controller" in note_text(messages):
            pass                       # resume after a preemption: the framework already rolled back derived writes; the policy repairs from the note/refreshed read
        elif _tls.new_note and has_retract and "rejected" not in note_text(messages):
            # ideal MTPO agent strategy: retract everything done so far, then redo the task on the refreshed view
            self.seen_notes = n
            k = _my_writes_since_restart(messages)
            if k:
                return Turn("", [{"id": f"call_{self.n}_r", "name": "nexus__retract_my_actions", "args": {"count": k}}], Usage(0, 0, 0, 0, 1, 0.0))
            _tls.new_note = False   # nothing written yet: fall through to the policy with the truncated history
        calls = self.policy(messages)
        self.seen_notes = n
        if calls is None:
            return Turn("Done.", [], Usage(0, 0, 0, 0, 1, 0.0))
        tcs = [{"id": f"call_{self.n}_{i}", "name": name.replace(".", "__"), "args": args} for i, (name, args) in enumerate(calls)]
        return Turn("", tcs, Usage(0, 0, 0, 0, 1, 0.0))
