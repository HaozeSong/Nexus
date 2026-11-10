# Engine-side plugin: content-keyed sampling in SGLang

Nexus needs two things from the inference engine beyond what any server already offers (request abort, prefix caching,
telemetry):

1. **Content-keyed sampling.** The engine's deterministic sampler draws Gumbel noise from
   `murmur_hash32(seed, position, token)`. `content_keyed_sampler.py` registers a sampling backend `nexus` whose sampler uses a
   *content key* in place of the position: the MurmurHash3 fold of the last *m* token ids of the request's sequence and the number
   of earlier occurrences of that window. The same PRF is implemented in pure Python in `nexus/sampling.py`, so the client-side
   verifier and the engine draw identical noise.
2. **Draft verification.** No engine change: the client scores a chunk of *b* draft tokens with one prefill request through
   the logprob interface (`return_logprob`, `logprob_start_len`, `top_logprobs_num`) and applies the keyed rule at each position
   (`nexus/engine/sglang_native.py`, `_score_chunk`).

## Install

```bash
# in the SGLang environment, with this repository on PYTHONPATH
python -m nexus.engine.sglang_plugin.launch --model-path <model> --tp 2 --port 30001 --sampling-backend nexus \
    --tool-call-parser <parser> --reasoning-parser <parser>
```

`launch.py` imports the plugin (which calls `register_sampler_backend("nexus", ...)` and wraps
`SamplingBatchInfo.from_schedule_batch` to attach per-request keys and seeds) and then runs `sglang.launch_server` unchanged.
Requests opt in through `sampling_params.custom_params = {"nexus_seed", "nexus_window", "nexus_position_keys"}`; other requests
are sampled as before. Written against SGLang 0.5.x (`sglang/srt/layers/sampler.py` with `register_sampler_backend`,
`multinomial_with_seed(logprobs, seed, positions)`).

## Status

`tests/test_sampling.py` checks the plugin's PRF, keys and acceptance rule against the client-side sampler, and the model-backed
experiments run the server with it (`--keyed-engine` in the drivers). Without the plugin, the client
still verifies drafts with keyed sampling while ordinary decoding uses the engine's own sampler; that mode is exact at
temperature 0 and approximate otherwise (`nexus/engine/sglang_native.py` module docstring).
