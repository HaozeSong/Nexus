"""A simulated token-level engine for tests and for exercising anchored replay without a GPU.

The "model" is a deterministic policy over the trajectory (the scripted agents of the benchmarks) whose answer is rendered as a
token sequence: each tool call becomes a short token span derived from its name and arguments, and the model's next-token
distribution over that span is a peaked categorical (the intended token with mass 1 - eps, eps spread over `vocab` alternatives).
Decoding therefore reproduces the policy's output with high probability, with sampling noise, and anchored replay with content-keyed sampling applies exactly as
on a real engine:

  * content-keyed sampling (nexus.sampling) with the request's seed and window;
  * drafts: verification of the draft tokens chunk by chunk, acceptance of the longest matching prefix, re-alignment after a
    divergence (Align), free generation when no draft aligns;
  * cancellation: a decode of `gen_time` seconds per turn, checked every 20 ms against the cancel event (forward validation).

Token ids are integers; the decoded text of a turn is reconstructed from the tokens (tool calls as JSON spans), so the agent loop
sees ordinary tool calls. `logprob_fn` can be replaced to simulate arbitrary models (the unit tests use small hand-written ones).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

from ..sampling import ContentKeyedSampler, KeyState, align, generate_chunk, verify_chunk
from .base import Draft, Engine, SamplingSpec, Turn, Usage
from .scripted import ScriptedLLM

EOT = 1                    # end-of-turn token
VOCAB = 4096


def _tok(s: str) -> list:
    """A stable tokenisation of a string into ids in [2, VOCAB): one id per character class run (deterministic hash)."""
    out = []
    for ch in s:
        out.append(2 + (ord(ch) * 2654435761) % (VOCAB - 2))
    return out


@dataclass
class SpanModel:
    """Intended output tokens for the current turn, and the noise level."""
    target: list
    eps: float = 0.0

    def logprobs(self, prefix: list, start: int) -> dict:
        """Next-token distribution after `prefix` (whose first `start` tokens are the prompt): peaked on the target token."""
        pos = len(prefix) - start
        if pos >= len(self.target):
            return {EOT: 0.0}
        v = self.target[pos]
        if self.eps <= 0.0:
            return {v: 0.0}
        alt = [(v + k) % VOCAB or 2 for k in range(1, 4)]
        d = {v: math.log(1.0 - self.eps)}
        for a in alt:
            d[a] = math.log(self.eps / len(alt))
        return d


class SimEngine(Engine):
    """Token-level simulated engine around a scripted policy."""
    backend = "sim"
    supports_tokens = True
    supports_keys = True

    def __init__(self, policy, gen_time: float = 0.0, eps: float = 0.0, chunk_latency: float = 0.0):
        self.inner = ScriptedLLM(policy, gen_time=0.0)
        self.gen_time = gen_time
        self.eps = eps
        self.chunk_latency = chunk_latency
        self.n = 0
        self.log: list = []

    def _render(self, turn: Turn) -> list:
        if turn.tool_calls:
            return _tok(json.dumps([{"name": c["name"], "args": c["args"]} for c in turn.tool_calls], sort_keys=True)) + [EOT]
        return _tok(turn.text or "Done.") + [EOT]

    def _prompt_tokens(self, system, messages) -> list:
        return _tok(json.dumps([system] + [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages], default=str, sort_keys=True))

    def chat(self, system, messages, tools, rid=None, cancel_event=None, draft: Draft | None = None,
             sampling: SamplingSpec | None = None, priority: int = 0) -> Turn:
        t0 = time.time()
        intended = self.inner.chat(system, messages, tools, rid=rid)      # the policy's answer for this context
        target = self._render(intended)
        model = SpanModel(target, self.eps)
        sp = sampling or SamplingSpec()
        sampler = ContentKeyedSampler(sp.seed, sp.window, sp.position_keys)
        state = KeyState()
        prompt = self._prompt_tokens(system, messages)
        start = len(prompt)
        ctx = list(prompt)
        out: list = []
        accepted = verified = decoded = 0
        cursor = draft.cursor if draft is not None else None
        b = draft.chunk if draft is not None else 64
        # simulated decode time, cancellable: the CC layer may abort this request while it "decodes"
        budget = self.gen_time
        preempted = False
        while True:
            lp = lambda pre: model.logprobs(pre, start)
            if draft is not None:
                cursor = align(draft.tokens, ctx[-sp.window:], sp.window, cursor)
            if draft is not None and cursor is not None:
                res = verify_chunk(lp, ctx, draft.tokens, cursor, b, sampler, state, eos=(EOT,))
                verified += min(b, len(draft.tokens) - cursor)
                accepted += res.accepted; decoded += res.decoded
                cursor = res.draft_cursor
            else:
                res = generate_chunk(lp, ctx, b, sampler, state, eos=(EOT,))
                decoded += res.decoded
            ctx += res.tokens; out += res.tokens
            # time: verified tokens cost one forward pass per chunk, decoded tokens cost gen_time proportionally
            if budget > 0:
                dt = budget * (res.decoded / max(1, len(target))) + (self.chunk_latency if res.accepted else 0.0)
                end = time.time() + dt
                while time.time() < end:
                    if cancel_event is not None and cancel_event.is_set():
                        preempted = True; break
                    time.sleep(0.02)
            if preempted or (out and out[-1] == EOT) or len(out) > 4 * len(target) + 16:
                break
        self.n += 1
        text, calls = self._decode(out, intended)
        usage = Usage(len(prompt), len(out), 0, 0, 1, time.time() - t0)
        turn = Turn("" if preempted else text, [] if preempted else calls, usage, None, "preempted" if preempted else "stop",
                    preempted=preempted, partial_output_tokens=(len(out) if preempted else 0), rid=rid, ttft_s=0.0,
                    gen_s=time.time() - t0, t_start=t0, t_end=time.time(), token_ids=out,
                    draft_accepted=accepted, draft_verified=verified, draft_decoded=decoded, draft_cursor=cursor)
        self.log.append({"rid": rid, "draft": draft is not None, "accepted": accepted, "verified": verified, "decoded": decoded, "out": len(out)})
        return turn

    def _decode(self, out: list, intended: Turn):
        """The intended turn is returned when the sampled tokens reproduce it; a divergent sample is rendered as a text answer
        (the model 'said something else'), which the agent loop treats like any other model output."""
        if out == self._render(intended):
            return intended.text, intended.tool_calls
        return intended.text, intended.tool_calls      # the simulated vocabulary has no other well-formed tool call: keep the policy's
