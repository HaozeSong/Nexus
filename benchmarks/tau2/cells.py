"""Contended tau2-airline cells: five templates instantiated on N_RES reservations (2 passengers, credit card
on file, both flights available, alternatives with seats).  The first instance of a template carries the template's name; further
instances add _2, _3, ..."""
from __future__ import annotations

import json

from nexus.cell import Cell
from benchmarks.tau2.backend import PRISTINE, Tau2AirlineBackend
from tau2.domains.airline.tools import AirlineTools

SYSTEM_EXTRA = ("You are an airline customer-service agent operating the reservation system through the airline tools. "
                "Reservation ids are 6-character codes; flight numbers look like 'HAT123'; dates are YYYY-MM-DD. "
                "update_reservation_* tools replace the WHOLE list you pass (passengers / flights), so always fetch the "
                "reservation first and pass the complete updated list. Never modify a reservation whose status is "
                "'cancelled'. Do not ask for confirmation; act directly.")
N_RES = 40
CELLS: dict[str, Cell] = {}
PARAMS: dict[str, dict] = {}
_tools = AirlineTools(PRISTINE)


def _alt(leg, cabin):
    for f in _tools.search_direct_flight(leg[2], leg[3], leg[1]):
        if f.flight_number != leg[0] and f.available_seats[cabin] >= 2:
            return f.flight_number
    return None


def _candidates():
    out, users = [], set()
    for rid, res in PRISTINE.reservations.items():
        u = PRISTINE.users[res.user_id]
        cards = [p for p in u.payment_methods.values() if p.source == "credit_card"]
        if len(res.passengers) != 2 or not cards or res.insurance != "no" or getattr(res, "status", None) == "cancelled" or len(res.flights) != 2:
            continue
        if res.user_id in users:
            continue
        if not all(PRISTINE.flights[f.flight_number].dates.get(f.date) and PRISTINE.flights[f.flight_number].dates[f.date].status == "available" for f in res.flights):
            continue
        legs = [(f.flight_number, f.date, f.origin, f.destination) for f in res.flights]
        alts = [_alt(l, res.cabin) for l in legs]
        if any(a is None for a in alts):
            continue
        users.add(res.user_id)
        out.append(dict(R=rid, USER=res.user_id, CARD=cards[0].id, PAX=[(p.first_name, p.last_name, p.dob) for p in res.passengers],
                        FLIGHTS=legs, ALT=alts, BAGS=(res.total_baggages, res.nonfree_baggages),
                        PAID=sum(p.amount for p in res.payment_history), CABIN=res.cabin))
    return out


_cands = _candidates()
_cands = [c for c in _cands if c["R"] == "JP6LYC"] + [c for c in _cands if c["R"] != "JP6LYC"]
_cands = _cands[:N_RES]


def _r(store, R):
    return json.loads(store.read("airline.get_reservation_details", {"reservation_id": R}))


def _paid(rec):
    return sum(p["amount"] for p in rec["payment_history"])


def _cert(store, user):
    """Certificates added relative to the base state (multiset difference)."""
    u = store.record("users", user)
    now = sorted(p["amount"] for p in u["payment_methods"].values() if p.get("source") == "certificate")
    base = sorted(p.amount for p in PRISTINE.users[user].payment_methods.values() if p.source == "certificate")
    for b in base:
        if b in now:
            now.remove(b)
    return now


def _add(name, template, cell, params):
    cell.backend = Tau2AirlineBackend; cell.benchmark = "tau2"; cell.system_extra = SYSTEM_EXTRA
    if "R" in params:
        cell.objects = {f"res:{params['R']}"}   # cells are keyed by their reservation
    cell.template = template
    CELLS[name] = cell; PARAMS[name] = dict(params, template=template)


def _name(base, i):
    return base if i == 0 else f"{base}_{i+1}"


for i, C in enumerate(_cands):
    R, USER, CARD, PAX, FLIGHTS, ALT, BAGS, PAID, CABIN = (C[k] for k in ("R", "USER", "CARD", "PAX", "FLIGHTS", "ALT", "BAGS", "PAID", "CABIN"))
    new_dob = PAX[0][2][:-2] + ("27" if PAX[0][2][-2:] != "27" else "28")
    new_last = PAX[1][1] + "ini" if not PAX[1][1].endswith("ini") else PAX[1][1] + "s"

    def inv_pax(store, R=R, PAX=PAX, new_dob=new_dob, new_last=new_last):
        rec = _r(store, R)
        pax = [(p["first_name"], p["last_name"], p["dob"]) for p in rec["passengers"]]
        want = [(PAX[0][0], PAX[0][1], new_dob), (PAX[1][0], new_last, PAX[1][2])]
        if rec.get("status") == "cancelled":
            return False, "reservation cancelled"
        return (True, "both passenger fixes present") if pax == want else (False, f"passengers {pax} (lost update)")
    _add(_name("passengers_lost_update", i), "passengers_lost_update", Cell(_name("passengers_lost_update", i), "lost update: two whole-list rewrites of the passenger list",
        task_a=f"In reservation {R}, passenger {PAX[0][0]} {PAX[0][1]}'s date of birth is wrong: correct it to {new_dob}. Keep every other passenger detail unchanged.",
        task_b=f"In reservation {R}, the last name of passenger {PAX[1][0]} {PAX[1][1]} is misspelled: it should be '{new_last}'. Keep every other passenger detail unchanged.",
        invariant=inv_pax), dict(C, new_dob=new_dob, new_last=new_last))

    def inv_bags(store, R=R, BAGS=BAGS, PAID=PAID):
        rec = _r(store, R)
        ok = rec["total_baggages"] == BAGS[0] + 3 and rec["nonfree_baggages"] == BAGS[1] + 2 and abs(_paid(rec) - (PAID + 100)) < 1
        return (True, "both baggage changes present, $100 charged") if ok else (False, f"bags {rec['total_baggages']}/{rec['nonfree_baggages']} paid {_paid(rec)} (base {BAGS}, {PAID})")
    _add(_name("baggage_lost_update", i), "baggage_lost_update", Cell(_name("baggage_lost_update", i), "lost update: two absolute rewrites of baggage counts",
        task_a=f"Add one more checked bag to reservation {R}. It is a free bag (within the cabin allowance), so the number of non-free bags stays the same; no payment is needed (use payment id {CARD} for the tool call).",
        task_b=f"Add two more NON-free checked bags to reservation {R}, charging the fees to payment method {CARD}. Keep the free bags as they are.",
        invariant=inv_bags), dict(C))

    def inv_fl(store, R=R, ALT=ALT, CABIN=CABIN):
        rec = _r(store, R)
        fl = [f["flight_number"] for f in rec["flights"]]
        return (True, "both legs changed") if fl == ALT and rec["cabin"] == CABIN else (False, f"flights {fl}, expected {ALT}")
    _add(_name("flights_lost_update", i), "flights_lost_update", Cell(_name("flights_lost_update", i), "lost update: two whole-list rewrites of the flight list",
        task_a=f"Change the first flight of reservation {R} from {FLIGHTS[0][0]} on {FLIGHTS[0][1]} to {ALT[0]} on the same date, same cabin ({CABIN}). Keep the second flight exactly as it is. Charge any fare difference to {CARD}.",
        task_b=f"Change the second flight of reservation {R} from {FLIGHTS[1][0]} on {FLIGHTS[1][1]} to {ALT[1]} on the same date, same cabin ({CABIN}). Keep the first flight exactly as it is. Charge any fare difference to {CARD}.",
        invariant=inv_fl), dict(C))

    def inv_cb(store, R=R):
        rec = _r(store, R)
        if rec["status"] != "cancelled":
            return False, f"status {rec['status']}"
        return (False, f"net payments {_paid(rec)} != 0 (charge on a cancelled reservation not refunded)") if abs(_paid(rec)) > 0.5 else (True, "cancelled, fully refunded")
    _add(_name("cancel_vs_bags", i), "cancel_vs_bags", Cell(_name("cancel_vs_bags", i), "stale status read: modification charged after cancellation",
        task_a=f"Cancel reservation {R} entirely.",
        task_b=f"Add two more NON-free checked bags to reservation {R}, charging the fees to payment method {CARD}. If the reservation is cancelled, do nothing.",
        invariant=inv_cb), dict(C))

    def inv_cc(store, R=R, USER=USER, BAGS=BAGS, PAID=PAID):
        rec = _r(store, R); certs = _cert(store, USER)
        if rec["status"] != "cancelled":
            return False, f"status {rec['status']}"
        if abs(_paid(rec)) > 0.5:
            return False, f"net payments {_paid(rec)} != 0"
        if len(certs) != 1:
            return False, f"expected one certificate, got {certs}"
        bags = (rec["total_baggages"], rec["nonfree_baggages"])
        if bags == BAGS and abs(certs[0] - PAID) < 1:
            return True, "A·B: cancelled + certificate for original total; bags untouched"
        if bags == (BAGS[0] + 2, BAGS[1] + 2) and abs(certs[0] - (PAID + 100)) < 1:
            return True, "B·A: bags charged, certificate covers the higher total"
        return False, f"bags {bags} certificate {certs[0]} (base bags {BAGS}, paid {PAID})"
    _add(_name("cancel_certificate_vs_bags", i), "cancel_certificate_vs_bags", Cell(_name("cancel_certificate_vs_bags", i), "canary + unrecoverable effect: certificate amount derived from a total another agent changes",
        task_a=f"The airline has cancelled the flights of reservation {R}. Cancel the reservation, then send its user {USER} a certificate whose amount equals the total amount they had paid for this reservation (sum of its payment history before cancellation).",
        task_b=f"Add two more NON-free checked bags to reservation {R}, charging the fees to payment method {CARD}. If the reservation is cancelled, do nothing.",
        invariant=inv_cc), dict(C))
