"""Content-keyed sampling.

An LLM samples each token from p(v | x). Under Gumbel-max sampling the choice is

    v* = argmax_v [ log p(v | x) + g_v ],      g_v ~ Gumbel(0, 1) i.i.d.

Fixing the noise g_v makes the sample reproducible for a given prefix while leaving the distribution intact. Nexus derives the
noise from a pseudo-random function keyed by the *content* of the preceding context instead of by the token position:

    u_v = PRF(seed_T, w, n, v),   g_v = -log(-log u_v),

where w is the window of the last m tokens and n counts earlier occurrences of that window in the trajectory. Across the original
execution and an anchored replay, matching windows (and occurrence counts) produce the same noise, so the randomness re-aligns after a
local change even when absolute positions differ. A position-keyed sampler (the engine's deterministic mode) is kept as the
ablation baseline.

Verification and ordinary decoding use the same sampler: at a draft position, verification computes the keyed sample for the updated
prefix and accepts the draft token only if the two agree; at the first mismatch the keyed sample is emitted and decoding continues
(equivalence of verification and decoding). The accepted tokens are therefore exactly the prefix ordinary decoding would have
produced.

The PRF is MurmurHash3 (x86, 32-bit) over the four 32-bit blocks (seed_lo, seed_hi, key, token) with length finalisation 16, which is
bit-for-bit the hash the SGLang engine uses for deterministic sampling (`sglang.kernels.ops.sampling.murmur_hash`); the engine-side
plugin (nexus/engine/sglang_plugin) substitutes the content key for the position in that hash, so an engine with the plugin and this
module draw identical noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

MASK32 = 0xFFFFFFFF
UINT32_MAX = 0xFFFFFFFF
_C1, _C2 = 0xCC9E2D51, 0x1B873593
_LOG_CAP = -(2.0 ** -32)          # the engine clamps log(u) to <= -2^-32 so that u == 1 does not give +inf Gumbel noise
_LOG_MIN = -1.7976931348623157e308


def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & MASK32


def _fmix32(h: int) -> int:
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & MASK32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & MASK32
    h ^= h >> 16
    return h


def _mix(h: int, k: int) -> int:
    k = (k * _C1) & MASK32
    k = _rotl32(k, 15)
    k = (k * _C2) & MASK32
    h ^= k
    h = _rotl32(h, 13)
    return (h * 5 + 0xE6546B64) & MASK32


def murmur3_32(seed: int, key: int, token: int) -> int:
    """MurmurHash3_x86_32 over the blocks (seed_lo, seed_hi, key, token), finalised with len=16. Returns a uint32."""
    seed &= 0xFFFFFFFFFFFFFFFF
    h = 0
    h = _mix(h, seed & MASK32)
    h = _mix(h, (seed >> 32) & MASK32)
    h = _mix(h, key & MASK32)
    h = _mix(h, token & MASK32)
    h ^= 16
    return _fmix32(h)


def gumbel_noise(seed: int, key: int, token: int) -> float:
    """The engine's Gumbel noise for (seed, key, token): -log(-log u) with u = hash / UINT32_MAX and the engine's clamps."""
    u = murmur3_32(seed, key, token) / UINT32_MAX
    lu = math.log(u) if u > 0.0 else _LOG_MIN
    lu = min(lu, _LOG_CAP)
    return -math.log(-lu)


def window_key(window: Sequence[int], occurrence: int) -> int:
    """32-bit content key of a window of token ids and its occurrence count (w, n); folds the tokens through the same
    MurmurHash3 mixing so that every token of the window contributes."""
    h = 0
    for t in window:
        h = _mix(h, int(t) & MASK32)
    h = _mix(h, int(occurrence) & MASK32)
    h ^= 4 * (len(window) + 1)
    return _fmix32(h)


@dataclass
class KeyState:
    """Occurrence counts of windows seen so far in one trajectory (the n of (w, n)). A copy is taken at a savepoint and restored on
    rollback, so that a replay counts occurrences exactly as the original execution did up to the divergence."""
    counts: dict = field(default_factory=dict)

    def copy(self) -> "KeyState":
        return KeyState(dict(self.counts))


class ContentKeyedSampler:
    """Gumbel-max sampling with content-keyed noise (default) or position-keyed noise (`position_keys=True`, the ablation).

    `sample(candidates, key)` picks argmax over `candidates` of log p + g; `candidates` is a mapping token -> log-probability of the
    temperature-scaled distribution. With the full vocabulary this is exact; with the top-k candidates the engine returns through its
    logprob interface (nexus.engine.sglang_native) the sample is exact whenever the keyed argmax lies inside the top-k, which at the
    temperatures used (<= 1.0) is the case for all but a negligible mass; the in-engine plugin runs the same rule on the full vocabulary.
    """

    def __init__(self, seed: int, m: int = 8, position_keys: bool = False):
        self.seed = int(seed)
        self.m = int(m)
        self.position_keys = position_keys

    # -- keys -----------------------------------------------------------------------------------------------------
    def key(self, prefix: Sequence[int], state: KeyState, position: int | None = None) -> int:
        """The sampling key for the next token after `prefix` (its last m tokens and their occurrence count so far). `state` is read
        but not updated; call `note` after the token is chosen."""
        if self.position_keys:
            return (len(prefix) if position is None else int(position)) & MASK32
        w = tuple(int(t) for t in prefix[-self.m:])
        return window_key(w, state.counts.get(w, 0))

    def note(self, prefix: Sequence[int], state: KeyState) -> None:
        """Record that the window ending at `prefix` has been used once more (after a token was sampled at its position)."""
        if self.position_keys:
            return
        w = tuple(int(t) for t in prefix[-self.m:])
        state.counts[w] = state.counts.get(w, 0) + 1

    # -- sampling -------------------------------------------------------------------------------------------------
    def sample(self, candidates: Mapping[int, float], key: int) -> int:
        """argmax_v [log p(v) + g_v] over the candidates, with g_v = Gumbel noise keyed by (seed, key, v)."""
        best, best_score = None, -math.inf
        for v, lp in candidates.items():
            if lp == -math.inf:
                continue
            s = lp + gumbel_noise(self.seed, key, int(v))
            if s > best_score or (s == best_score and best is not None and int(v) < best):
                best, best_score = int(v), s
        if best is None:
            best = int(max(candidates, key=candidates.get))
        return best

    def greedy(self, candidates: Mapping[int, float]) -> int:
        return int(max(candidates, key=candidates.get))


# -- draft alignment (Align) -----------------------------------------------------------------------------
def align(draft: Sequence[int], context_tail: Sequence[int], m: int, cursor: int | None) -> int | None:
    """Choose where in the draft D the replay should continue from, given the last m tokens of the current context.

    1. Right after a substituted token (cursor = index in D following the token that diverged): try the old continuation at that
       index, i.e. return `cursor`.
    2. Otherwise search D for the longest suffix of `context_tail` (down to one token) and continue after the last match of it
       (reference-based alignment, Chen et al. 2023 / Yang et al. 2023); None if no alignment is found and the model must generate
       without a draft until one is.
    """
    if not draft:
        return None
    if cursor is not None and 0 <= cursor < len(draft):
        return cursor
    tail = [int(t) for t in context_tail[-m:]]
    for n in range(min(len(tail), m), 0, -1):
        suffix = tail[-n:]
        pos = _rfind(draft, suffix)
        if pos is not None:
            nxt = pos + n
            return nxt if nxt < len(draft) else None
    return None


def _rfind(seq: Sequence[int], sub: Sequence[int]) -> int | None:
    n = len(sub)
    for i in range(len(seq) - n, -1, -1):
        if list(seq[i:i + n]) == list(sub):
            return i
    return None


# -- reference verify-and-generate loop (used by the simulated engine and the unit tests) ---------------------------------
@dataclass
class VerifyResult:
    tokens: list              # tokens appended to the context by this step
    accepted: int             # draft tokens accepted (verified)
    decoded: int              # tokens produced by the keyed sampler itself (the first differing sample, or free generation)
    draft_cursor: int | None  # where the draft continues after this step (None: realign)


def verify_chunk(logprob_fn, prefix: list, draft: Sequence[int], start: int, b: int, sampler: ContentKeyedSampler,
                 state: KeyState, eos: Iterable[int] = ()) -> VerifyResult:
    """Verify up to b draft tokens D[start:start+b] against the model for `prefix`.

    `logprob_fn(prefix) -> Mapping[token, logp]` is the model's next-token distribution (temperature applied). A real engine
    computes all b positions in one forward pass; the reference implementation calls the model per position, which is only a
    matter of cost, not of the tokens produced: both accept the longest prefix of the draft that the keyed sampler reproduces and
    append the first differing keyed sample.
    """
    eos = set(int(e) for e in eos)
    out, accepted = [], 0
    ctx = list(prefix)
    for j in range(start, min(len(draft), start + b)):
        key = sampler.key(ctx, state)
        v = sampler.sample(logprob_fn(ctx), key)
        sampler.note(ctx, state)
        out.append(v); ctx.append(v)
        if v == int(draft[j]):
            accepted += 1
            if v in eos:
                return VerifyResult(out, accepted, 0, None)
            continue
        # first mismatch: the keyed sample replaces the draft token; the draft continues after the substituted position (Align rule 1)
        return VerifyResult(out, accepted, 1, j + 1)
    return VerifyResult(out, accepted, 0, start + len(out))


def generate_chunk(logprob_fn, prefix: list, b: int, sampler: ContentKeyedSampler, state: KeyState, eos: Iterable[int] = ()) -> VerifyResult:
    """Decode up to b tokens with the keyed sampler and no draft."""
    eos = set(int(e) for e in eos)
    out = []
    ctx = list(prefix)
    for _ in range(b):
        key = sampler.key(ctx, state)
        v = sampler.sample(logprob_fn(ctx), key)
        sampler.note(ctx, state)
        out.append(v); ctx.append(v)
        if v in eos:
            break
    return VerifyResult(out, 0, len(out), None)
