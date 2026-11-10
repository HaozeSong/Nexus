"""Concurrency-control protocols, selected by name.

A protocol lives in `protocols/<name>/protocol.py` as a class `P` (a `nexus.transaction.Protocol` subclass). The Nexus
ablations are small enough to share one module, `protocols/nexus/ablations.py`.
"""
from __future__ import annotations

import importlib
from pathlib import Path

BASELINES = ("serial", "naive", "twopl", "occ", "mtpo", "sbus", "atomix", "cordon", "memtx")


def load_protocol(name: str):
    """The protocol class registered under `name`."""
    from protocols.nexus.ablations import ABLATIONS
    if name in ABLATIONS:
        return ABLATIONS[name]
    try:
        return importlib.import_module(f"protocols.{name}.protocol").P
    except ModuleNotFoundError as e:
        if e.name in (f"protocols.{name}", f"protocols.{name}.protocol"):
            raise KeyError(f"unknown protocol {name!r}; available: {', '.join(available())}") from None
        raise      # the protocol exists but one of its own imports is missing (e.g. an unfetched third_party repository)


def available() -> list[str]:
    """Every selectable protocol name: the packages of this directory, then the Nexus ablations."""
    from protocols.nexus.ablations import ABLATIONS
    here = Path(__file__).parent
    packages = sorted(p.parent.name for p in here.glob("*/protocol.py"))
    ordered = [n for n in (*BASELINES, "nexus") if n in packages] + [n for n in packages if n not in (*BASELINES, "nexus")]
    return ordered + list(ABLATIONS)
