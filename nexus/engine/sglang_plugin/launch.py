"""Launch an SGLang server with the Nexus content-keyed sampling backend installed.

    python -m nexus.engine.sglang_plugin.launch --model-path <model> --sampling-backend nexus [other sglang.launch_server args]

Importing nexus.engine.sglang_plugin.content_keyed_sampler registers the "nexus" sampling backend and wraps the sampling-batch
construction with per-request content keys; the rest is sglang.launch_server unchanged. The client (nexus.engine.sglang_native)
opts a request in by sending nexus_seed / nexus_window / nexus_position_keys in sampling_params.custom_params.
"""
from __future__ import annotations

import sys


def main():
    from . import content_keyed_sampler  # noqa: F401  (registers the backend on import)
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    if "--sampling-backend" not in sys.argv:
        sys.argv += ["--sampling-backend", "nexus"]
    server_args = prepare_server_args(sys.argv[1:])
    launch_server(server_args)


if __name__ == "__main__":
    main()
