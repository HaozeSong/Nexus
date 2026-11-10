"""Footprint declarations (every tool registers its read set R(τ) and write set W(τ)) and the conflict relation.

Objects: "table:id" for one record, "table:*" for the table as a range (a listing / range read, or an update of
records not known before execution), "table:+" for an INSERT (a record created; its id is known only from the result,
and the observed write set names it). Two objects conflict when they name the same table and either is the range, or
both are the same id. An insert conflicts with the range (a listing may now be missing the new record: the phantom) but
not with any existing record and not with another insert.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Callable


# arguments that carry content derived from what the agent read (as opposed to identifying a record)
CONTENT_ARGS = {"content", "new_text", "old_text", "body", "text", "notes", "new_value", "passengers", "flights", "items"}


def is_content_arg(key: str, value) -> bool:
    """Whether a tool argument carries derived content: a known content parameter, a long string, or a structure."""
    return key in CONTENT_ARGS or (isinstance(value, str) and len(value) > 40) or isinstance(value, (list, dict))


def conflicts(a: str, b: str) -> bool:
    ta, ia = a.split(":", 1)
    tb, ib = b.split(":", 1)
    return ta == tb and (ia == "*" or ib == "*" or (ia == ib and ia != "+"))


def table_level(obj: str) -> bool:
    """A range/table-level object: the range itself or an insert under it (both hit a listing; only the range hits records)."""
    return obj.endswith(":*") or obj.endswith(":+")


def footprint_conflict(writes: set[str], other: set[str]) -> set[str]:
    return {w for w in writes if any(conflicts(w, o) for o in other)}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: str                       # "read" | "write"
    table: str                      # primary table the tool acts on
    id_param: str | None = None     # parameter carrying the record id (None for range/create)
    range_write: bool = False       # create/send: an insert into the table (phantom source for listings: t:+)
    unrecoverable: bool = False     # no inverse: held until lower-sigma agents commit (MTPO)
    footprint: Callable | None = None   # optional custom (kwargs, result, alloc_ids) -> (R, W)
    id_col: str | None = None       # id column in JSON results of range reads

    def declared(self, kwargs: dict, lookup: Callable | None = None) -> tuple[set[str], set[str]]:
        """Footprint readable off the invocation (before execution). `lookup(table, rid)` (the caller's view of the store) lets a
        footprint resolve the records a tool will touch through the database rather than declaring a whole table."""
        if self.footprint:
            if lookup is not None and len(inspect.signature(self.footprint).parameters) >= 4:
                return self.footprint(kwargs, None, [], lookup)
            return self.footprint(kwargs, None, [])
        t = self.table
        if self.kind == "read":
            if self.id_param and kwargs.get(self.id_param):
                return {f"{t}:{kwargs[self.id_param]}"}, set()
            return {f"{t}:*"}, set()
        if self.range_write:
            return set(), {f"{t}:+"}          # a create: conflicts with listings only; the new id is added by observed()
        rid = kwargs.get(self.id_param) if self.id_param else None
        obj = f"{t}:{rid}" if rid else f"{t}:*"
        return {obj}, {obj}

    def observed(self, kwargs: dict, result: str, alloc_ids: list[str]) -> tuple[set[str], set[str]]:
        """Footprint after execution: range reads add the returned ids, creates add the new id."""
        if self.footprint:
            return self.footprint(kwargs, result, alloc_ids)
        r, w = self.declared(kwargs)
        t = self.table
        if self.kind == "read" and not (self.id_param and kwargs.get(self.id_param)) and self.id_col:
            try:
                rows = json.loads(result)
                if isinstance(rows, list):
                    r |= {f"{t}:{row[self.id_col]}" for row in rows if isinstance(row, dict) and self.id_col in row}
            except (json.JSONDecodeError, TypeError):
                pass
        if self.kind == "write":
            w |= {f"{t}:{i}" for i in alloc_ids}
        return r, w


def schemas_from(descriptions: dict[str, tuple[str, dict]], style: str, names=None) -> list[dict]:
    """descriptions: name -> (description, json-schema parameters). Wire names use '__' for '.'."""
    out = []
    for n in names or list(descriptions):
        desc, params = descriptions[n]
        wire = n.replace(".", "__")
        if style == "anthropic":
            out.append({"name": wire, "description": desc[:1500], "input_schema": params})
        else:
            out.append({"type": "function", "function": {"name": wire, "description": desc[:1500], "parameters": params}})
    return out


def wire_to_tool(name: str) -> str:
    return name.replace("__", ".")
