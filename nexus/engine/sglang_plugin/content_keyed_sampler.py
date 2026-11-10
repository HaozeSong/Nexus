"""Content-keyed sampling inside SGLang: the engine-side half of nexus.sampling.

SGLang's deterministic sampler draws Gumbel noise with murmur_hash32(seed, position, token) (sglang.srt.layers.sampler,
`multinomial_with_seed`). This plugin registers a sampling backend "nexus" whose sampler replaces the position by a *content key*:
the MurmurHash3 fold of the last m token ids of the request's sequence (prompt + output so far) and the number of earlier
occurrences of that window in the sequence (nexus.sampling.window_key). Requests opt in through `sampling_params.custom_params`
{"nexus_seed": int, "nexus_window": m, "nexus_position_keys": bool}; other requests are sampled exactly as before.

Two pieces are needed:

  1. the sampler (this module, ~60 lines of engine logic): `NexusSampler` overrides `_sample_from_probs` to use the content keys as
     the `positions` argument of `multinomial_with_seed`, which keeps the engine's own hash kernel and all of its numerics;
  2. per-request keys: the sampler only sees the batch, so `install()` wraps `SamplingBatchInfo.from_schedule_batch` to compute one
     key per request from `req.origin_input_ids + req.output_ids` (occurrence counts are maintained incrementally on the request
     object) and to stash the per-request seeds. Requests without the opt-in get their ordinary position and seed.

Install by launching the server through `python -m nexus.engine.sglang_plugin.launch --sampling-backend nexus <sglang args>`
(it imports this module, which registers the backend and installs the wrapper, then hands over to sglang.launch_server), or by
importing this module in any process that constructs the model runner before `create_sampler` runs. tests/test_sampling.py checks
the PRF and the keys against the client-side sampler.

Draft verification does not need engine changes: the client scores a draft chunk with one prefill request through the logprob
interface (nexus.engine.sglang_native._score_chunk) and applies the same keyed rule; with this plugin installed, verified and
decoded tokens follow the same distribution and the same noise (equivalence of verification and decoding).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MASK32 = 0xFFFFFFFF


def _rotl32(x, r):
    return ((x << r) | (x >> (32 - r))) & MASK32


def _fmix32(h):
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & MASK32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & MASK32
    h ^= h >> 16
    return h


def _mix(h, k):
    k = (k * 0xCC9E2D51) & MASK32
    k = _rotl32(k, 15)
    k = (k * 0x1B873593) & MASK32
    h ^= k
    h = _rotl32(h, 13)
    return (h * 5 + 0xE6546B64) & MASK32


def window_key(window, occurrence: int) -> int:
    """Identical to nexus.sampling.window_key."""
    h = 0
    for t in window:
        h = _mix(h, int(t) & MASK32)
    h = _mix(h, int(occurrence) & MASK32)
    h ^= 4 * (len(window) + 1)
    return _fmix32(h)


def request_key(req, m: int) -> int:
    """Content key for the next token of `req`: window = last m tokens of prompt+output, n = earlier occurrences in the sequence.
    Occurrence counts are kept on the request (`req._nexus_counts`) and advanced one token at a time; the first call counts the
    windows of the prompt."""
    seq = list(req.origin_input_ids) + list(req.output_ids)
    counts = getattr(req, "_nexus_counts", None)
    done = getattr(req, "_nexus_counted", 0)
    if counts is None:
        counts, done = {}, 0
    # count every window that ENDS before the current position (occurrences earlier in the trajectory)
    for end in range(max(done, 1), len(seq)):
        w = tuple(seq[max(0, end - m):end])
        counts[w] = counts.get(w, 0) + 1
    req._nexus_counts, req._nexus_counted = counts, max(done, len(seq))
    w = tuple(seq[-m:])
    return window_key(w, counts.get(w, 0))


def install():
    """Register the "nexus" sampling backend and wrap SamplingBatchInfo construction with per-request content keys."""
    import torch
    from sglang.srt.layers import sampler as _sampler
    from sglang.srt.sampling import sampling_batch_info as _sbi

    class NexusSampler(_sampler.Sampler):
        def _sample_from_probs(self, probs, sampling_info, positions, simple_sampling_case):
            keys = getattr(sampling_info, "nexus_keys", None)
            seeds = getattr(sampling_info, "nexus_seeds", None)
            if keys is None or seeds is None:
                return super()._sample_from_probs(probs, sampling_info, positions, simple_sampling_case)
            # content keys replace positions; seeds replace the engine's per-request sampling seed; everything else unchanged.
            # Requests that did not opt in keep their position and seed (the wrapper copies them in).
            logprobs = torch.log(probs.to(torch.float64))
            if not simple_sampling_case:
                # apply top-k / top-p / min-p filtering exactly as the pytorch fallback does, then sample with the keyed noise
                probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                probs_sort[torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1) >= sampling_info.top_ks.view(-1, 1)] = 0.0
                probs_sort[(probs_sum - probs_sort) > sampling_info.top_ps.view(-1, 1)] = 0.0
                if sampling_info.need_min_p_sampling:
                    probs_sort[probs_sort < (probs_sort[:, 0] * sampling_info.min_ps).view(-1, 1)] = 0.0
                lp = probs_sort.to(torch.float64).log_()
                idx = _sampler.multinomial_with_seed(lp, seeds, keys)
                return torch.gather(probs_idx.to(torch.int32), dim=1, index=idx).view(-1), None
            idx = _sampler.multinomial_with_seed(logprobs, seeds, keys)
            return idx.view(-1).to(torch.int32), None

    _sampler.register_sampler_backend("nexus", NexusSampler)

    orig = _sbi.SamplingBatchInfo.from_schedule_batch.__func__

    def from_schedule_batch(cls, batch, vocab_size):
        info = orig(cls, batch, vocab_size)
        reqs = batch.reqs
        keys, seeds, any_opt = [], [], False
        for r in reqs:
            cp = getattr(r.sampling_params, "custom_params", None) or {}
            if isinstance(cp, dict) and "nexus_seed" in cp:
                any_opt = True
                m = int(cp.get("nexus_window", 8))
                if cp.get("nexus_position_keys"):
                    keys.append(len(r.origin_input_ids) + len(r.output_ids))
                else:
                    keys.append(request_key(r, m))
                seeds.append(int(cp["nexus_seed"]) & 0x7FFFFFFFFFFFFFFF)
            else:
                keys.append(len(r.origin_input_ids) + len(r.output_ids))
                seeds.append(int(getattr(r.sampling_params, "sampling_seed", None) or 42))
        if any_opt:
            info.nexus_keys = torch.tensor(keys, dtype=torch.int64, device=batch.device)
            info.nexus_seeds = torch.tensor(seeds, dtype=torch.int64, device=batch.device)
        else:
            info.nexus_keys = None
            info.nexus_seeds = None
        return info

    _sbi.SamplingBatchInfo.from_schedule_batch = classmethod(from_schedule_batch)
    logger.info("nexus content-keyed sampler installed (sampling backend 'nexus')")


try:  # importing this module inside an SGLang process installs the plugin; elsewhere it is importable for its pure functions
    import sglang  # noqa: F401
    install()
except Exception:
    pass
