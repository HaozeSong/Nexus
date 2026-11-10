"""Backend = a deterministic, copyable state plus the tools that act on it.

The log-replay ObjectStore is backend-agnostic: it only needs (1) a fresh copy of the base state,
(2) `run_tool(state, op)` that executes one tool deterministically on that state (ids allocated through
`op.next_id(table)` so replays reuse the ids handed out at first execution), (3) per-table id columns for
record lookup, and (4) the footprint-declared ToolSpecs plus LLM schemas.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    name: str = "base"
    id_cols: dict[str, str] = {}
    specs: dict = {}                 # tool name -> ToolSpec

    @abstractmethod
    def fresh_base(self):
        """A fresh copy of the pristine state (called once per ObjectStore)."""

    @abstractmethod
    def copy(self, state):
        """Deep copy of a state (called per materialisation)."""

    @abstractmethod
    def run_tool(self, state, tool: str, kwargs: dict, op) -> str:
        """Execute `tool` on `state`; allocate ids through op.next_id(table). Returns the tool's string result."""

    @abstractmethod
    def initial_next_ids(self, base) -> dict[str, int]:
        """Per-table next id counters derived from the base state."""

    @abstractmethod
    def record(self, state, table: str, rid: str) -> dict | None:
        """One record as a dict (for notifications / invariants)."""

    @abstractmethod
    def tool_schemas(self, style: str, names=None) -> list[dict]:
        """LLM tool schemas: style 'anthropic' or 'openai'. Wire names use '__' instead of '.'."""

    def format_id(self, n: int) -> str:
        return str(n).zfill(8)

    def wire_to_tool(self, name: str) -> str:
        return name.replace("__", ".")

    # -- write effects (optional): what a write durably added, so a retracted write can be checked for having been redone ------
    def write_effect(self, before, after, tool: str, kwargs: dict):
        """The durable effect of a write as the backend understands it (e.g. the lines it added to a text object), computed from
        the object before and after the write in the writer's own view. None = unknown: the redo is then tracked by identity only."""
        return None

    def effect_holds(self, state, effect) -> bool:
        """Whether `effect` (from write_effect) is present in `state`."""
        return True

    def effect_brief(self, effect) -> str:
        return str(effect)
