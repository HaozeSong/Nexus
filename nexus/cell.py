"""Cell = one contended workload (agent-1 task + agent-2 task + invariant) on a benchmark backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Check = tuple[bool, str]


@dataclass
class Cell:
    name: str
    anomaly: str
    task_a: str
    task_b: str
    invariant: Callable
    backend: type | None = None
    benchmark: str = ""
    system_extra: str = ""
    objects: set = field(default_factory=set)   # records the two tasks touch (the workload driver pairs record-disjoint cells)
    group: str = ""                             # suite the cell was loaded from (benchmarks/registry.py)
    template: str = ""                          # conflict template the cell instantiates (short suites); "" = its own template

    @property
    def qualified(self):
        return f"{self.benchmark}/{self.name}"
