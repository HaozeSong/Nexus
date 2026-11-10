"""tau2-bench airline domain (sierra-research/tau2-bench, MIT) as a Backend: JSON database (flights with
per-date seat inventory, users with payment methods, reservations) driven through the benchmark's own
AirlineTools.  Tools are the official ones; footprints are declared here."""
from __future__ import annotations

import copy
import json
import os
import re
import threading

from nexus.backend import Backend
from nexus.paths import THIRD_PARTY, vendored
from nexus.tools import ToolSpec, schemas_from

# tau2 reads its data directory from the environment when it is imported, so the default is set here, before the import
os.environ.setdefault("TAU2_DATA_DIR", str(THIRD_PARTY / "tau2-bench" / "data"))
vendored("tau2-bench", "src")

from loguru import logger  # noqa: E402

logger.remove()

from tau2.domains.airline.data_model import FlightDB  # noqa: E402
from tau2.domains.airline.tools import AirlineTools  # noqa: E402
from tau2.domains.airline.utils import AIRLINE_DB_PATH  # noqa: E402

PRISTINE: FlightDB = FlightDB.load(AIRLINE_DB_PATH)
_tls = threading.local()


def _new_reservation_id(self):
    return _tls.op.next_id("reservations")


AirlineTools._get_new_reservation_id = _new_reservation_id   # stable ids across replays

_TOOL_OBJS = AirlineTools(copy.deepcopy(PRISTINE)).get_tools()
TOOL_NAMES = [n for n in _TOOL_OBJS if n not in ("transfer_to_human_agents",)]
_DESC = {f"airline.{n}": (t.openai_schema["function"]["description"], t.openai_schema["function"]["parameters"]) for n, t in _TOOL_OBJS.items()}


def _res(k):
    return f"reservations:{k['reservation_id']}"


def _user(k):
    return f"users:{k['user_id']}"


def _fp_book(k, r, alloc):
    fl = {f"flights:{f.get('flight_number')}" for f in (k.get("flights") or []) if isinstance(f, dict)}
    return ({_user(k)} | fl, {_user(k), "reservations:+"} | fl | {f"reservations:{i}" for i in alloc})


def _owner(r, k=None, lookup=None):
    """users:<owner> of the reservation: from the tool result (the reservation JSON) after execution, from the caller's snapshot of
    the reservation before it; users:* only when
    the reservation is unknown."""
    if r and isinstance(r, str):
        m = re.search(r'"user_id"\s*:\s*"([^"]+)"', r)
        if m:
            return f"users:{m.group(1)}"
    if lookup is not None and k is not None and k.get("reservation_id"):
        try:
            rec = lookup("reservations", k["reservation_id"])
        except Exception:
            rec = None
        if isinstance(rec, dict) and rec.get("user_id"):
            return f"users:{rec['user_id']}"
    return "users:*"


def _fp_update_flights(k, r, alloc, lookup=None):
    """Declared footprint of a flight change: the reservation, its owner and the legs the call actually CHANGES. A leg that is already
    on the reservation in the same cabin is not re-booked, so it is not a premise of the write.
    Before execution (`r is None`) the current legs come from the caller's snapshot via `lookup`; after execution the write set is
    every listed leg (conservative)."""
    new = {f.get("flight_number") for f in (k.get("flights") or []) if isinstance(f, dict)}
    if r is None and lookup is not None and k.get("reservation_id"):
        try:
            rec = lookup("reservations", k["reservation_id"])
        except Exception:
            rec = None
        if isinstance(rec, dict) and (k.get("cabin") in (None, rec.get("cabin"))):
            new -= {f.get("flight_number") for f in (rec.get("flights") or []) if isinstance(f, dict)}
    fl = {f"flights:{x}" for x in new if x}
    u = _owner(r, k, lookup)
    return ({_res(k), u} | fl, {_res(k), u} | fl)


def _fp_update_bags(k, r, alloc, lookup=None):
    u = _owner(r, k, lookup)
    return ({_res(k), u}, {_res(k), u})


def _fp_search(k, r, a):
    """A flight search is a listing (flights:*) that also SHOWS the records it returns: those are re-observed individually, so an
    invalidated entry for a flight the search lists is refreshed."""
    return ({"flights:*"} | {f"flights:{x}" for x in re.findall(r'"flight_number"\s*:\s*"([A-Z0-9]+)"', r or "")}, set())


SPECS = {s.name: s for s in [
    ToolSpec("airline.calculate", "read", "none", footprint=lambda k, r, a: (set(), set())),
    ToolSpec("airline.list_all_airports", "read", "airports"),
    ToolSpec("airline.search_direct_flight", "read", "flights", footprint=_fp_search),
    ToolSpec("airline.search_onestop_flight", "read", "flights", footprint=_fp_search),
    ToolSpec("airline.get_flight_status", "read", "flights", "flight_number"),
    ToolSpec("airline.get_user_details", "read", "users", footprint=lambda k, r, a: ({_user(k)} | ({f"reservations:{x}" for x in re.findall(r'"([A-Z0-9]{6})"', r or "")} or {"reservations:*"}), set())),
    ToolSpec("airline.get_reservation_details", "read", "reservations", "reservation_id"),
    ToolSpec("airline.book_reservation", "write", "reservations", footprint=_fp_book),
    ToolSpec("airline.cancel_reservation", "write", "reservations", footprint=lambda k, r, a: ({_res(k)}, {_res(k)})),
    ToolSpec("airline.update_reservation_baggages", "write", "reservations", footprint=_fp_update_bags),
    ToolSpec("airline.update_reservation_passengers", "write", "reservations", footprint=lambda k, r, a: ({_res(k)}, {_res(k)})),
    ToolSpec("airline.update_reservation_flights", "write", "reservations", footprint=_fp_update_flights),
    ToolSpec("airline.send_certificate", "write", "users", footprint=lambda k, r, a: ({_user(k)}, {_user(k)}), unrecoverable=True),
]}
# The payment-charging updates also write the reservation's owner (payment history). The OBSERVED footprint names the owner from the
# returned reservation JSON; the DECLARED footprint (locks, admission, premise check) resolves it from the caller's snapshot of the
# reservation (`lookup`), and says users:* only for an unknown reservation: declaring users:* for every call would make 2PL lock
# every user and admission wait for every older user write.


def _jsonable(o):
    if hasattr(o, "model_dump_json"):
        return json.loads(o.model_dump_json())
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(i) for i in o]
    return o


def _to_str(x):
    if hasattr(x, "model_dump_json"):
        return x.model_dump_json()
    if isinstance(x, (list, tuple, dict)):
        return json.dumps(_jsonable(x), default=str)      # nested pydantic objects (e.g. one-stop itineraries) included
    return str(x)


class Tau2AirlineBackend(Backend):
    name = "tau2"
    id_cols = {"reservations": "reservation_id", "users": "user_id", "flights": "flight_number"}
    specs = SPECS

    def fresh_base(self):
        return copy.deepcopy(PRISTINE)

    def copy(self, state):
        return copy.deepcopy(state)

    def run_tool(self, state, tool, kwargs, op):
        _tls.op = op
        try:
            fn = getattr(AirlineTools(state), tool.split(".", 1)[1])
            return _to_str(fn(**kwargs))
        except Exception as e:  # tools raise ValueError on policy/availability errors
            return f"Error: {e}"
        finally:
            _tls.op = None

    def initial_next_ids(self, base):
        return {"reservations": 1}

    def format_id(self, n):
        return f"NEW{n:03d}"

    def record(self, state, table, rid):
        coll = getattr(state, table, None)
        if coll is None or rid not in coll:
            return None
        return json.loads(coll[rid].model_dump_json())

    def tool_schemas(self, style, names=None):
        return schemas_from(_DESC, style, [f"airline.{n}" for n in (names or TOOL_NAMES)])
