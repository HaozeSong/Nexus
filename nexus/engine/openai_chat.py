"""OpenAI-compatible chat engine (any vLLM / SGLang / OpenAI-style server) and an Anthropic Messages path.

Capabilities (nexus.engine.base.Engine): streaming with cancellation (the CC layer aborts a request through SGLang's
/abort_request); no token ids and no drafts, so anchored replay degrades to regeneration without drafts on this engine.  Internal message format:
  {"role":"user","content":str}
  {"role":"assistant","content":str,"tool_calls":[{"id","name","args"}],"raw":...}
  {"role":"tool","tool_call_id":str,"content":str}
"""
from __future__ import annotations

import json
import os
import time

from .base import Draft, Engine, SamplingSpec, Turn, Usage  # noqa: E402,F401


class LLM(Engine):
    supports_tokens = False
    supports_keys = False

    def __init__(self, model: str, backend: str = "openai", base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0, max_tokens: int = 2048,
                 extra: dict | None = None, stream: bool = True):
        self.model, self.backend, self.temperature, self.max_tokens = model, backend, temperature, max_tokens
        self.extra = extra or {}
        self.stream = stream
        self.base_url = base_url
        self.root_url = base_url.rstrip("/")[:-3] if base_url and base_url.rstrip("/").endswith("/v1") else base_url
        if backend == "anthropic":   # credentials from the environment (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL)
            import anthropic
            kw = {}
            if base_url or os.environ.get("ANTHROPIC_BASE_URL"):
                kw["base_url"] = base_url or os.environ["ANTHROPIC_BASE_URL"]
            if api_key:
                kw["api_key"] = api_key
            elif os.environ.get("ANTHROPIC_AUTH_TOKEN") and not os.environ.get("ANTHROPIC_API_KEY"):
                kw["auth_token"] = os.environ["ANTHROPIC_AUTH_TOKEN"]
            self.client = anthropic.Anthropic(max_retries=4, timeout=180, **kw)
        elif backend == "openai":
            import openai
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                                        max_retries=4, timeout=300)
        else:
            raise ValueError(backend)

    # ------------------------------------------------------------------ anthropic
    def _anth_messages(self, messages):
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no output)"}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def _chat_anthropic(self, system, messages, tools) -> Turn:
        t0 = time.monotonic()
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=self._anth_messages(messages), tools=tools, **self.extra)
        text, calls = "", []
        for b in resp.content:
            if b.type == "text":
                text += b.text
            elif b.type == "tool_use":
                calls.append({"id": b.id, "name": b.name, "args": dict(b.input)})
        u = resp.usage
        usage = Usage(u.input_tokens, u.output_tokens, getattr(u, "cache_read_input_tokens", 0) or 0,
                      getattr(u, "cache_creation_input_tokens", 0) or 0, 1, time.monotonic() - t0)
        return Turn(text, calls, usage, resp, resp.stop_reason)

    # ------------------------------------------------------------------ openai-compatible
    def _oai_messages(self, system, messages):
        out = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                          "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                                         for tc in m["tool_calls"]]
                if m.get("reasoning_content"):
                    msg["reasoning_content"] = m["reasoning_content"]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        return out

    def abort(self, rid: str):
        """Ask the engine to stop decoding this request immediately (SGLang /abort_request)."""
        import urllib.request
        try:
            req = urllib.request.Request(f"{self.root_url}/abort_request", data=json.dumps({"rid": rid}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10).read()
            return True
        except Exception:
            return False

    def _chat_openai_stream(self, system, messages, tools, rid, cancel_event) -> Turn:
        t0 = time.monotonic(); t_wall0 = time.time()
        extra = dict(self.extra)
        body = {"rid": rid} if rid else {}
        stream = self.client.chat.completions.create(
            model=self.model, messages=self._oai_messages(system, messages), tools=tools,
            temperature=self.temperature, max_tokens=self.max_tokens, stream=True,
            stream_options={"include_usage": True}, extra_body=body, **extra)
        text, reasoning, calls, usage, finish, ttft, n_chunks, preempted = "", "", {}, None, "", 0.0, 0, False
        for chunk in stream:
            if cancel_event is not None and cancel_event.is_set():
                preempted = True
                if rid:
                    self.abort(rid)
                try:
                    stream.close()
                except Exception:
                    pass
                break
            if getattr(chunk, "usage", None):
                u = chunk.usage
                cached = 0
                try:
                    cached = u.prompt_tokens_details.cached_tokens or 0
                except AttributeError:
                    pass
                usage = Usage(u.prompt_tokens - cached, u.completion_tokens, cached, 0, 1, 0.0)
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            if chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
            if d is None:
                continue
            if ttft == 0.0 and (d.content or getattr(d, "reasoning_content", None) or d.tool_calls):
                ttft = time.monotonic() - t0
            if d.content:
                text += d.content; n_chunks += 1
            rc = getattr(d, "reasoning_content", None)
            if rc:
                reasoning += rc; n_chunks += 1
            for tc in d.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": tc.id or f"call_{rid}_{tc.index}", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments; n_chunks += 1
        gen_s = time.monotonic() - t0
        parsed = []
        for i in sorted(calls):
            c = calls[i]
            try:
                args = json.loads(c["args"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": c["args"]}
            parsed.append({"id": c["id"], "name": c["name"], "args": args})
        if usage is None:
            usage = Usage(0, n_chunks, 0, 0, 1, 0.0)   # aborted: engine sends no usage; count streamed chunks
        usage.latency_s = gen_s
        t = Turn(text, [] if preempted else parsed, usage, None, "preempted" if preempted else finish,
                 preempted=preempted, partial_output_tokens=(n_chunks if preempted else 0), rid=rid,
                 ttft_s=ttft, gen_s=gen_s, t_start=t_wall0, t_end=time.time())
        t.reasoning = reasoning or None
        return t

    def _chat_openai(self, system, messages, tools) -> Turn:
        t0 = time.monotonic()
        resp = self.client.chat.completions.create(
            model=self.model, messages=self._oai_messages(system, messages), tools=tools,
            temperature=self.temperature, max_tokens=self.max_tokens, **self.extra)
        ch = resp.choices[0]
        calls = []
        for tc in ch.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append({"id": tc.id, "name": tc.function.name, "args": args})
        u = resp.usage
        cached = 0
        try:
            cached = u.prompt_tokens_details.cached_tokens or 0
        except AttributeError:
            pass
        usage = Usage(u.prompt_tokens - cached, u.completion_tokens, cached, 0, 1, time.monotonic() - t0)
        t = Turn(ch.message.content or "", calls, usage, resp, ch.finish_reason or "")
        t.reasoning = getattr(ch.message, "reasoning_content", None)
        return t

    def chat(self, system: str, messages: list[dict], tools: list[dict], rid: str | None = None,
             cancel_event=None, draft: Draft | None = None, sampling: SamplingSpec | None = None, priority: int = 0) -> Turn:
        if self.backend == "anthropic":
            return self._chat_anthropic(system, messages, tools)
        if self.stream:
            return self._chat_openai_stream(system, messages, tools, rid, cancel_event)
        t = self._chat_openai(system, messages, tools)
        t.rid = rid
        return t
