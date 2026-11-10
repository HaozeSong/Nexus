"""MTPO (Monotonic Trajectory Pre-Order), the CoAgent protocol (arXiv:2606.15376), as a configuration of the
speculative substrate (nexus/speculative.py): launch-order ranks, filtered reads, speculative writes with late-write UNDO of
the higher-rank suffix (Thomas rule for same-field blind writes), one-way notifications riding on tool returns, retract tool,
unrecoverable holds and GlobalQuiet commit. protocols/mtpo/README.md maps each rule of the published protocol to this implementation."""
from __future__ import annotations

from nexus.speculative import Speculative


class P(Speculative):
    name = "mtpo"
    RETRACT_TOOL = True
    LATE_WRITE = "undo"
