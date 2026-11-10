"""Prompt construction and tool-call parsing for the token-level engine client.

The token-level client (nexus.engine.sglang_native) must build the prompt itself (the engine's /generate endpoint takes token ids)
and parse the model's tool calls from raw output text (the chat endpoint's parsers are not available on that path). Both are
model-family specific; two families (MiniMax, Hermes) are covered, and a generic JSON format as a fallback:

  * ``minimax``: MiniMax-M2 style ``<minimax:tool_call><invoke name="t"><parameter name="k">v</parameter>...</invoke></minimax:tool_call>``
    with optional ``<think>...</think>`` reasoning;
  * ``hermes``: ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` (Qwen / DeepSeek-style JSON tool calls);
  * ``json``: a bare JSON object or list of ``{"name", "arguments"}`` objects anywhere in the text.

Prompts are rendered with the tokenizer's chat template (``transformers``), passing the tool schemas so that the template emits the
family's tool-definition block; assistant turns with tool calls and tool results are rendered in the OpenAI message format the
templates expect.
"""
from __future__ import annotations

import json
import re
import uuid

_MM_CALL = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.S)
_MM_INVOKE = re.compile(r'<invoke name="([^"]+)">(.*?)</invoke>', re.S)
_MM_PARAM = re.compile(r'<parameter name="([^"]+)">(.*?)</parameter>', re.S)
_HERMES = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def _coerce(v: str):
    v = v.strip()
    try:
        return json.loads(v)
    except Exception:
        return v


def parse_tool_calls(text: str, fmt: str = "minimax") -> tuple[str, list, str | None]:
    """Split raw model output into (visible text, tool calls, reasoning)."""
    reasoning = None
    m = _THINK.search(text)
    if m:
        reasoning = m.group(1).strip()
        text = text[:m.start()] + text[m.end():]
    calls = []
    if fmt == "minimax":
        for block in _MM_CALL.findall(text):
            for name, body in _MM_INVOKE.findall(block):
                args = {k: _coerce(v) for k, v in _MM_PARAM.findall(body)}
                calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "name": name, "args": args})
        text = _MM_CALL.sub("", text)
    elif fmt == "hermes":
        for js in _HERMES.findall(text):
            try:
                o = json.loads(js)
                calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "name": o["name"], "args": o.get("arguments", o.get("args", {})) or {}})
            except Exception:
                continue
        text = _HERMES.sub("", text)
    else:
        for js in re.findall(r"\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"[^{}]*\{[^{}]*\}[^{}]*\}", text, re.S):
            try:
                o = json.loads(js)
                calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "name": o["name"], "args": o.get("arguments", o.get("args", {})) or {}})
            except Exception:
                continue
    return text.strip(), calls, reasoning


def to_openai_messages(system: str, messages: list) -> list:
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                      "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}} for tc in m["tool_calls"]]
            if m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
    return out


class PromptBuilder:
    """Renders (system, messages, tools) to token ids with the served model's chat template."""

    def __init__(self, tokenizer_path: str, trust_remote_code: bool = True):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
        self.eos_ids = {i for i in [self.tok.eos_token_id] if i is not None}
        for name in ("<|im_end|>", "<|end_of_sentence|>", "</s>", "[e~[", "<|endoftext|>"):
            try:
                i = self.tok.convert_tokens_to_ids(name)
                if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                    self.eos_ids.add(i)
            except Exception:
                pass

    def build(self, system: str, messages: list, tools: list) -> list:
        msgs = to_openai_messages(system, messages)
        try:
            ids = self.tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, tokenize=True)
        except TypeError:
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return list(ids)

    def decode(self, ids: list) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)
