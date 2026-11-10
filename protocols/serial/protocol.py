"""Serial: agents run back-to-back on the live state (the reference execution)."""
from nexus.transaction import Protocol


class P(Protocol):
    name = "serial"
    SEQUENTIAL = True   # the runner executes agents one after another
