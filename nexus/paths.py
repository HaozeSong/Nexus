"""Repository paths and access to the vendored repositories under third_party/.

The benchmarks (tau2-bench, WorkBench) and two baselines (atomix, MEMTX_) reuse their authors' code, which is cloned by
scripts/fetch_third_party.sh and is not pip-installable as a package of its own. `vendored()` is the one place that puts
such a source tree on sys.path; a module that needs one calls it before importing from it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"


def vendored(repo: str, *sub: str) -> Path:
    """Path of `third_party/<repo>/<sub...>`, made importable (inserted at the front of sys.path once)."""
    path = THIRD_PARTY.joinpath(repo, *sub)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path
