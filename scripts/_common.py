"""Shared CLI pieces for the experiment scripts: engine construction and result-file resume."""
from __future__ import annotations

import argparse

from nexus.paths import ROOT  # noqa: F401  (re-exported for the drivers)


def add_engine_args(ap: argparse.ArgumentParser):
    ap.add_argument("--engine", default="sglang", choices=["sglang", "openai", "sim"],
                    help="sglang: token-level client (drafts, keyed sampling); openai: any OpenAI-compatible chat server "
                         "(cancellation only; replay degrades to regeneration without drafts); sim: simulated engine, no model")
    ap.add_argument("--base-url", default="http://127.0.0.1:30001/v1")
    ap.add_argument("--model", default="MiniMax-M2.7")
    ap.add_argument("--tokenizer", default=None, help="tokenizer path or HF id for --engine sglang (default: --model)")
    ap.add_argument("--tool-format", default="minimax", choices=["minimax", "hermes", "json"])
    ap.add_argument("--keyed-engine", action="store_true", help="the server runs the nexus sampling backend (engine plugin)")
    ap.add_argument("--logprobs-scaled", action="store_true", help="the server returns temperature-scaled logprobs")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--verify-topk", type=int, default=32)


def root_url(base_url: str) -> str:
    b = base_url.rstrip("/")
    return b[:-3] if b.endswith("/v1") else b


def engine_factory(a):
    """Returns factory(role_or_agent) -> Engine, and starts the telemetry sampler for server engines."""
    if a.engine == "sim":
        raise SystemExit("--engine sim needs a scripted policy per cell; use scripts/scripted_test.py or the unit tests")
    from nexus.runner import start_sampler
    start_sampler(root_url(a.base_url))
    if a.engine == "openai":
        from nexus.engine.openai_chat import LLM
        return lambda who: LLM(a.model, "openai", a.base_url, "EMPTY", a.temperature, a.max_tokens, extra={"top_p": a.top_p})
    from nexus.engine.sglang_native import SGLangNative
    return lambda who: SGLangNative(a.base_url, a.model, a.tokenizer or a.model, tool_format=a.tool_format, max_tokens=a.max_tokens,
                                    temperature=a.temperature, top_p=a.top_p, verify_topk=a.verify_topk,
                                    keyed_engine=a.keyed_engine, logprobs_scaled=a.logprobs_scaled)


def engine_ok(base_url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(root_url(base_url) + "/health", timeout=10).read()
        return True
    except Exception:
        return False


def protocol_overrides(p, overrides: dict | None):
    """Return a subclass of protocol class `p` with class attributes overridden (e.g. KEY_WINDOW=16 for a sensitivity sweep)."""
    if not overrides:
        return p
    return type(p.__name__, (p,), dict(overrides, name=p.name + "".join(f"_{k.lower()}{v}" for k, v in overrides.items())))
