"""Token-level SGLang client: cancellation, drafts and content-keyed sampling against the engine's native /generate interface.

What it does per turn (nexus.engine.base.Engine.chat):

  * builds the prompt token ids with the model's chat template (PromptBuilder) and streams a generation with `rid` so that the CC
    layer can cancel it through POST /abort_request (forward validation); the partial output tokens of a cancelled request
    are returned (they become part of the draft);
  * with a draft D (anchored replay), runs the verify-and-generate loop: a chunk of b draft tokens is scored by
    one prefill request (input_ids = context + chunk, return_logprob with logprob_start_len = len(context), top_logprobs_num = k);
    the content-keyed sampler (nexus.sampling) is applied at every draft position on the returned candidate distributions, the
    longest matching prefix is accepted, the first differing keyed sample is appended, and the draft is re-aligned (Align); when no
    draft aligns, up to b tokens are decoded;
  * keyed decoding: with the engine plugin installed (nexus.engine.sglang_plugin, sampling backend "nexus") the request carries the
    seed / window / key mode in `custom_params` and the engine draws the same keyed noise as the client-side verifier, so verified
    and decoded tokens are exactly the tokens ordinary decoding produces. Without the plugin, decoding uses the engine's own
    sampler and verification is still keyed: the accepted prefix is then the one the keyed sampler reproduces (exact at temperature
    0, an approximation otherwise), which is the documented degraded mode.

Candidate distributions come from the engine's top-k logprob interface; the keyed argmax over the top-k equals the full-vocabulary
argmax whenever the sampled token lies in the top-k, so k=32 (the default) is exact for all but a negligible mass at the
temperatures used. Whether the engine returns raw or temperature-scaled logprobs is a server setting; `logprobs_scaled` tells the
client which, since dividing by the temperature a second time would bias verification.
"""
from __future__ import annotations

import json
import time
import urllib.request

from ..sampling import ContentKeyedSampler, KeyState, align
from .base import Draft, Engine, SamplingSpec, Turn, Usage
from .toolcalls import PromptBuilder, parse_tool_calls


class SGLangNative(Engine):
    backend = "sglang"
    supports_tokens = True
    supports_keys = True

    def __init__(self, base_url: str, model: str, tokenizer: str, tool_format: str = "minimax", max_tokens: int = 8192,
                 temperature: float = 1.0, top_p: float = 0.95, top_k: int = -1, verify_topk: int = 32, keyed_engine: bool = False,
                 logprobs_scaled: bool = False, timeout: float = 600.0):
        self.root = base_url.rstrip("/")
        if self.root.endswith("/v1"):
            self.root = self.root[:-3]
        self.model, self.max_tokens, self.temperature, self.top_p, self.top_k = model, max_tokens, temperature, top_p, top_k
        self.verify_topk, self.keyed_engine, self.logprobs_scaled, self.timeout = verify_topk, keyed_engine, logprobs_scaled, timeout
        self.tool_format = tool_format
        self.prompt = PromptBuilder(tokenizer)
        self.eos = set(self.prompt.eos_ids)

    # -- HTTP ---------------------------------------------------------------------------------------------------------
    def _post(self, path: str, body: dict, stream: bool = False):
        req = urllib.request.Request(self.root + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        return urllib.request.urlopen(req, timeout=self.timeout)

    def abort(self, rid: str) -> bool:
        try:
            self._post("/abort_request", {"rid": rid}).read()
            return True
        except Exception:
            return False

    def _sampling_params(self, sampling: SamplingSpec | None, max_new: int, greedy: bool = False) -> dict:
        sp = {"max_new_tokens": max_new, "temperature": 0.0 if greedy else self.temperature, "top_p": 1.0 if greedy else self.top_p,
              "top_k": self.top_k, "skip_special_tokens": False}
        if sampling is not None and self.keyed_engine:
            sp["custom_params"] = {"nexus_seed": int(sampling.seed), "nexus_window": int(sampling.window),
                                   "nexus_position_keys": bool(sampling.position_keys)}
            sp["sampling_seed"] = int(sampling.seed) & 0x7FFFFFFF
        return sp

    # -- streaming decode ----------------------------------------------------------------------------------------------
    def _decode_stream(self, ids: list, rid: str, cancel_event, sampling: SamplingSpec | None, priority: int, max_new: int):
        """Stream a generation; returns (output token ids, finish reason, preempted, ttft, usage prompt stats)."""
        body = {"input_ids": ids, "sampling_params": self._sampling_params(sampling, max_new), "rid": rid, "stream": True,
                "return_logprob": True, "top_logprobs_num": 0, "logprob_start_len": len(ids)}
        if priority:
            body["priority"] = int(priority)
        t0 = time.monotonic()
        out, finish, preempted, ttft, cached = [], "", False, 0.0, 0
        resp = self._post("/generate", body, stream=True)
        try:
            for raw in resp:
                if cancel_event is not None and cancel_event.is_set():
                    preempted = True
                    self.abort(rid)
                    break
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                meta = ev.get("meta_info") or {}
                toks = meta.get("output_token_logprobs") or []
                if toks and ttft == 0.0:
                    ttft = time.monotonic() - t0
                # SGLang streams the cumulative list in some versions and the increment in others
                new = [int(t[1]) for t in toks]
                if len(new) >= len(out) and new[:len(out)] == out:
                    out = new
                else:
                    out += new
                if meta.get("finish_reason"):
                    finish = (meta["finish_reason"] or {}).get("type", "") if isinstance(meta.get("finish_reason"), dict) else str(meta.get("finish_reason"))
                cached = meta.get("cached_tokens", cached) or cached
        finally:
            try:
                resp.close()
            except Exception:
                pass
        return out, finish, preempted, ttft, cached

    # -- one prefill request scores a draft chunk --------------------------------------------------------------------------
    def _score_chunk(self, ids: list, chunk: list, sampling: SamplingSpec | None) -> tuple[list, dict]:
        """Candidate distributions at every position of `chunk` given `ids`, plus the next-token candidates after the chunk."""
        body = {"input_ids": ids + chunk, "sampling_params": self._sampling_params(sampling, 1), "stream": False,
                "return_logprob": True, "top_logprobs_num": self.verify_topk, "logprob_start_len": len(ids)}
        ev = json.loads(self._post("/generate", body).read())
        meta = ev.get("meta_info") or {}
        per_pos = []
        for pos in (meta.get("input_top_logprobs") or []):
            per_pos.append({int(t[1]): float(t[0]) for t in (pos or []) if t and t[0] is not None})
        nxt = {}
        otl = meta.get("output_top_logprobs") or []
        if otl:
            nxt = {int(t[1]): float(t[0]) for t in (otl[0] or []) if t and t[0] is not None}
        # the draft token's own logprob is always among the candidates (it may lie outside the top-k)
        for j, t in enumerate(meta.get("input_token_logprobs") or []):
            if j < len(per_pos) and t and t[0] is not None:
                per_pos[j].setdefault(int(t[1]), float(t[0]))
        return per_pos, nxt

    def _scaled(self, cands: dict) -> dict:
        if self.logprobs_scaled or self.temperature in (0.0, 1.0):
            return cands
        return {v: lp / self.temperature for v, lp in cands.items()}

    # -- the turn -----------------------------------------------------------------------------------------------------
    def chat(self, system, messages, tools, rid=None, cancel_event=None, draft: Draft | None = None,
             sampling: SamplingSpec | None = None, priority: int = 0) -> Turn:
        t_wall0 = time.time(); t0 = time.monotonic()
        ids = self.prompt.build(system, messages, tools)
        sp = sampling or SamplingSpec(temperature=self.temperature)
        sampler = ContentKeyedSampler(sp.seed, sp.window, sp.position_keys)
        state = KeyState()
        out: list = []
        accepted = verified = decoded = 0
        preempted, finish, ttft, cached = False, "", 0.0, 0
        cursor = draft.cursor if draft is not None else None
        b = draft.chunk if draft is not None else 64
        greedy = self.temperature == 0.0
        if draft is not None and draft.tokens:
            # verify-and-generate until the model emits end-of-turn or the draft is exhausted
            while len(out) < self.max_tokens:
                if cancel_event is not None and cancel_event.is_set():
                    preempted = True; break
                ctx = ids + out
                cursor = align(draft.tokens, ctx[-sp.window:], sp.window, cursor)
                if cursor is None:
                    break                      # no draft aligns: fall through to free decoding
                chunk = [int(t) for t in draft.tokens[cursor:cursor + b]]
                per_pos, nxt = self._score_chunk(ctx, chunk, sp)
                if not per_pos:
                    cursor = None; break
                verified += len(chunk)
                stop = False
                for j, dt in enumerate(chunk):
                    cands = self._scaled(per_pos[j]) if j < len(per_pos) else {}
                    if not cands:
                        stop = True; break
                    prefix = ids + out
                    key = sampler.key(prefix, state)
                    v = sampler.greedy(cands) if greedy else sampler.sample(cands, key)
                    sampler.note(prefix, state)
                    out.append(v)
                    if v == dt:
                        accepted += 1
                        if v in self.eos:
                            stop = True; cursor = None; finish = "stop"; break
                        continue
                    decoded += 1
                    cursor = cursor + j + 1        # Align rule 1: try the old continuation after the substituted token
                    if v in self.eos:
                        stop = True; cursor = None; finish = "stop"
                    break
                else:
                    # every draft token of the chunk accepted: the engine's next-token candidates give one more keyed sample
                    cursor += len(chunk)
                    if nxt and cursor >= len(draft.tokens):
                        prefix = ids + out
                        key = sampler.key(prefix, state)
                        v = sampler.greedy(self._scaled(nxt)) if greedy else sampler.sample(self._scaled(nxt), key)
                        sampler.note(prefix, state)
                        out.append(v); decoded += 1
                        if v in self.eos:
                            stop = True; cursor = None; finish = "stop"
                        else:
                            cursor = None
                if stop or (out and out[-1] in self.eos):
                    break
        if not preempted and not (out and out[-1] in self.eos):
            # ordinary decoding: the engine streams the rest of the turn
            more, finish2, pre2, ttft2, cached = self._decode_stream(ids + out, rid, cancel_event, sp, priority, self.max_tokens - len(out))
            out += more; decoded += len(more)
            preempted = preempted or pre2; finish = finish2 or finish; ttft = ttft2
        text = self.prompt.decode([t for t in out if t not in self.eos])
        visible, calls, reasoning = parse_tool_calls(text, self.tool_format)
        usage = Usage(len(ids) - int(cached or 0), len(out), int(cached or 0), 0, 1, time.monotonic() - t0)
        turn = Turn("" if preempted else visible, [] if preempted else calls, usage, None, "preempted" if preempted else (finish or "stop"),
                    preempted=preempted, partial_output_tokens=(len(out) if preempted else 0), rid=rid, ttft_s=ttft,
                    gen_s=time.monotonic() - t0, t_start=t_wall0, t_end=time.time(), token_ids=out,
                    draft_accepted=accepted, draft_verified=verified, draft_decoded=decoded, draft_cursor=cursor)
        turn.reasoning = reasoning
        return turn
