"""Naive: concurrent, no coordination.  Reads see the live state, writes land in place."""
from nexus.transaction import Protocol


class P(Protocol):
    name = "naive"
