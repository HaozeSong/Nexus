"""The inference-engine interface the agent loop and anchored replay are written against.

An engine turns a context (system prompt, messages, tool schemas) into one assistant turn. Three capabilities matter to Nexus:

  * cancellation: a request can be aborted while it decodes (forward validation);
  * drafts: a request can carry a draft continuation D and a sampling specification; the engine verifies draft tokens in chunks and
    re-decodes only where the keyed sample diverges (anchored replay); it reports how many tokens it accepted and decoded;
  * telemetry and priorities: queue depth / KV usage for the intent pricing policy, and a per-request priority.

Implementations: nexus.engine.sglang_native (token-level, drafts, keyed sampling through the engine plugin or through its logprob
interface), nexus.engine.openai_chat (any OpenAI-compatible server: cancellation only; replay degrades to regeneration without
drafts), nexus.engine.sim (a simulated engine for tests), nexus.engine.scripted (scripted agents, no model).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0
    latency_s: float = 0.0

    def add(self, o: "Usage"):
        self.input_tokens += o.input_tokens; self.output_tokens += o.output_tokens
        self.cache_read += o.cache_read; self.cache_write += o.cache_write
        self.calls += o.calls; self.latency_s += o.latency_s

    @property
    def total(self):
        return self.input_tokens + self.output_tokens + self.cache_read + self.cache_write

    def asdict(self):
        return {"input": self.input_tokens, "output": self.output_tokens, "cache_read": self.cache_read,
                "cache_write": self.cache_write, "calls": self.calls, "latency_s": round(self.latency_s, 3)}


@dataclass
class SamplingSpec:
    """Content-keyed sampling parameters: the task seed, the key window m, and whether to fall back to position keys."""
    seed: int = 0
    window: int = 8
    position_keys: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1


@dataclass
class Draft:
    """The invalidated suffix D of a trajectory, kept as a draft for anchored replay.

    `tokens` are the output tokens of the assistant turns after the replay point, in order, followed by the tokens of a cancelled
    generation; `cursor` is where the next verification should continue (None: realign by content). The engine advances the cursor
    as it consumes the draft and the agent loop carries it from turn to turn."""
    tokens: list = field(default_factory=list)
    cursor: int | None = 0
    chunk: int = 64                   # verification chunk b
    accepted: int = 0                 # bookkeeping across turns
    verified: int = 0
    decoded: int = 0

    def remaining(self) -> int:
        return 0 if self.cursor is None else max(0, len(self.tokens) - self.cursor)


@dataclass
class Turn:
    text: str
    tool_calls: list
    usage: Usage
    raw: object = None
    stop: str = ""
    preempted: bool = False          # the request was cancelled by the CC layer
    partial_output_tokens: int = 0   # tokens decoded before the cancellation
    rid: str | None = None
    ttft_s: float = 0.0
    gen_s: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0
    reasoning: str | None = None
    token_ids: list | None = None    # output token ids (incl. the partial output of a cancelled request) when the engine exposes them
    draft_accepted: int = 0          # draft tokens verified and kept
    draft_verified: int = 0          # draft positions examined
    draft_decoded: int = 0           # tokens produced by the sampler itself (first differing samples + free generation)
    draft_cursor: int | None = None  # where the draft continues after this turn


class Engine:
    """Base class; subclasses override `chat` and, if they can, `abort`."""
    backend = "base"
    supports_tokens = False          # exposes output token ids and accepts drafts
    supports_keys = False            # samples with content keys (engine plugin) or emulates them (logprob path)

    def chat(self, system: str, messages: list, tools: list, rid: str | None = None, cancel_event=None,
             draft: Draft | None = None, sampling: SamplingSpec | None = None, priority: int = 0) -> Turn:
        raise NotImplementedError

    def abort(self, rid: str) -> bool:
        return False

    def begin_task(self, task):  # optional hook used by the workload driver
        pass
