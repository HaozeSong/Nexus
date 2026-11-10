"""Long agentic transactions on the tau2-bench airline domain (main long-transaction benchmark, 2026-09-23).

Each task is a multi-request customer case on one real reservation (3–5 tool calls incl. reads; 2–3 writes with data
dependencies: the flight change determines the fare difference, the baggage allowance depends on the cabin, the booking
total on the flight prices). The concurrent partner acts on the same reservation. Correctness = the final state equals one of
the two serial executions of the two reference action lists (nexus.serial)."""
from __future__ import annotations

import json

from nexus.serial import serializable_cell
from benchmarks.tau2.backend import PRISTINE, Tau2AirlineBackend
from benchmarks.tau2.cells import SYSTEM_EXTRA, _cands

CELLS: dict = {}
SYSTEM_LONG = SYSTEM_EXTRA + (" A request may contain several changes: carry them out in the order given, one tool call per change, "
                              "re-reading the reservation after each change. Use exactly the payment method the request names.")


def _res_canon(rec):
    if rec is None:
        return None
    return (rec.get("status"), rec["cabin"], tuple((f["flight_number"], f["date"]) for f in rec["flights"]),
            tuple((p["first_name"], p["last_name"], p["dob"]) for p in rec["passengers"]),
            rec["total_baggages"], rec["nonfree_baggages"], rec["insurance"], round(sum(p["amount"] for p in rec["payment_history"])))


def _extract_factory(R, USER):
    be = Tau2AirlineBackend()

    def extract(state):
        rec = be.record(state, "reservations", R)
        user = be.record(state, "users", USER)
        certs = tuple(sorted(round(p["amount"]) for p in user["payment_methods"].values() if p.get("source") == "certificate"))
        new = tuple(sorted(_res_canon(json.loads(r.model_dump_json())) for rid, r in state.reservations.items()
                           if rid not in PRISTINE.reservations and r.user_id == USER))
        return (_res_canon(rec), certs, new)
    return extract


def _price(fn, date, cabin):
    return PRISTINE.flights[fn].dates[date].prices[cabin]


_TOOLS = []


def _search_tools():
    if not _TOOLS:
        import copy as _copy
        from benchmarks.tau2.backend import AirlineTools
        _TOOLS.append(AirlineTools(_copy.deepcopy(PRISTINE)))     # one shared read-only instance for all itinerary searches
    return _TOOLS[0]


def _itinerary(origin, destination, date, cabin, npax, avoid):
    """First direct or one-stop itinerary origin->destination on `date` with seats for npax in `cabin`, not using `avoid` flights."""
    tools = _search_tools()
    cands = [[f] for f in tools.search_direct_flight(origin, destination, date)]
    try:
        cands += [list(legs) for legs in tools.search_onestop_flight(origin, destination, date)]
    except Exception:
        pass
    for legs in cands:
        fns = [l.flight_number for l in legs]
        if set(fns) & set(avoid):
            continue
        ds = [getattr(l, "date", None) or date for l in legs]
        try:
            ok = all(PRISTINE.flights[fn].dates[d].available_seats.get(cabin, 0) >= npax for fn, d in zip(fns, ds))
        except KeyError:
            ok = False
        if ok:
            return [{"flight_number": fn, "date": d} for fn, d in zip(fns, ds)]
    return None


def _add(cell):
    exp = cell.expected()
    if any(e for _, _, e in exp):        # a reference replay hit a tool error (e.g. no seats): unusable case
        SKIPPED.append((cell.name, [e for _, _, e in exp if e]))
        return
    CELLS[cell.name] = cell


SKIPPED: list = []


def _cur(st, R):
    return json.loads(st.reservations[R].model_dump_json())


def _bags(R, d_total, d_nonfree, CARD):
    def f(st, be):
        r = _cur(st, R)
        if r.get("status") == "cancelled":
            return None          # policy: a cancelled reservation is not modified
        return ("airline.update_reservation_baggages", {"reservation_id": R, "total_baggages": r["total_baggages"] + d_total,
                                                        "nonfree_baggages": r["nonfree_baggages"] + d_nonfree, "payment_id": CARD})
    return f


def _pax(R, idx, field, value):
    def f(st, be):
        r = _cur(st, R)
        if r.get("status") == "cancelled":
            return None
        pax = [{"first_name": p["first_name"], "last_name": p["last_name"], "dob": p["dob"]} for p in r["passengers"]]
        pax[idx][field] = value
        return ("airline.update_reservation_passengers", {"reservation_id": R, "passengers": pax})
    return f


def _flights(R, cabin, replace: dict, CARD):
    """replace: leg index -> flight number (keeps the other legs and their dates as they currently are)."""
    def f(st, be):
        r = _cur(st, R)
        if r.get("status") == "cancelled":
            return None
        fl = [{"flight_number": replace.get(i, x["flight_number"]), "date": x["date"]} for i, x in enumerate(r["flights"])]
        return ("airline.update_reservation_flights", {"reservation_id": R, "cabin": cabin or r["cabin"], "flights": fl, "payment_id": CARD})
    return f


for i, C in enumerate(_cands):
    R, USER, CARD, PAX, FL, ALT, BAGS, PAID, CABIN = (C[k] for k in ("R", "USER", "CARD", "PAX", "FLIGHTS", "ALT", "BAGS", "PAID", "CABIN"))
    res = PRISTINE.reservations[R]
    dates = [f[1] for f in FL]
    pax = [{"first_name": p[0], "last_name": p[1], "dob": p[2]} for p in PAX]
    new_dob = PAX[0][2][:-2] + ("27" if PAX[0][2][-2:] != "27" else "28")
    new_last = PAX[1][1] + "ini" if not PAX[1][1].endswith("ini") else PAX[1][1] + "s"
    pax_fix1 = [dict(pax[0], dob=new_dob), pax[1]]
    pax_fix_both = [dict(pax[0], dob=new_dob), dict(pax[1], last_name=new_last)]
    alt_flights = [{"flight_number": ALT[0], "date": dates[0]}, {"flight_number": ALT[1], "date": dates[1]}]
    cur_flights = [{"flight_number": f[0], "date": f[1]} for f in FL]
    sfx = "" if i == 0 else f"_{i + 1}"
    objs = {f"res:{R}"}
    ex = _extract_factory(R, USER)
    up_cabin = "business" if CABIN != "business" else "economy"

    # ---- partner tasks (the concurrent agent) -------------------------------------------------------------------
    partner = {
        "bags": (f"Add one more checked bag to reservation {R}; it is a free bag within the allowance, so the number of non-free bags "
                 f"stays the same (use payment method {CARD} for the tool call).",
                 [_bags(R, 1, 0, CARD)]),
        "pax2": (f"In reservation {R}, the last name of passenger {PAX[1][0]} {PAX[1][1]} is misspelled: it should be '{new_last}'. "
                 f"Keep every other passenger detail unchanged.",
                 [_pax(R, 1, "last_name", new_last)]),
        "cancel": (f"Cancel reservation {R} entirely.", [("airline.cancel_reservation", {"reservation_id": R})]),
        "leg2": (f"Change the second flight of reservation {R} from {FL[1][0]} on {FL[1][1]} to {ALT[1]} on the same date, keeping the "
                 f"reservation's CURRENT cabin (read it from the reservation; do not change the cabin). Keep the first flight exactly as it "
                 f"is. Charge any fare difference to {CARD}.",
                 [_flights(R, None, {1: ALT[1]}, CARD)]),
    }
    pk = ["bags", "pax2", "cancel", "leg2"]

    # ---- LT1: rebook both legs, add bags, fix a passenger (3 dependent writes) ---------------------------------------
    ta = (f"Customer request for reservation {R} (user {USER}): (1) move both flights to {ALT[0]} and {ALT[1]} on the same dates as "
          f"now, same cabin ({CABIN}); charge any fare difference to {CARD} (refund to it if cheaper). (2) After that, add two more "
          f"NON-free checked bags, charging the fees to {CARD}: the total number of bags goes from {BAGS[0]} to {BAGS[0] + 2} and the "
          f"non-free count from {BAGS[1]} to {BAGS[1] + 2}. (3) Finally correct passenger "
          f"{PAX[0][0]} {PAX[0][1]}'s date of birth to {new_dob}, keeping every other passenger detail unchanged.")
    ref_a = [_flights(R, CABIN, {0: ALT[0], 1: ALT[1]}, CARD), _bags(R, 2, 2, CARD), _pax(R, 0, "dob", new_dob)]
    p = pk[i % 4]
    _add(serializable_cell(f"lt_rebook_bags_pax{sfx}", "tau2", Tau2AirlineBackend, ta, partner[p][0], ref_a, partner[p][1], ex, objs,
                           f"long: rebook+bags+passenger vs {p}", SYSTEM_LONG))

    # ---- LT2: cancel and rebook the same trip on a different, valid itinerary (cancel + booking with computed total) ----
    npax = len(pax)
    cur_fns = [f[0] for f in FL]
    out_it = _itinerary(res.origin, res.destination, FL[0][1], CABIN, npax, cur_fns)
    ret_it = None
    if res.flight_type == "round_trip":
        ret_it = _itinerary(res.destination, res.origin, FL[-1][1], CABIN, npax, cur_fns)
    new_fl = (out_it or []) + (ret_it or []) if out_it and (res.flight_type != "round_trip" or ret_it) else None
    if new_fl:
        total = sum(_price(f["flight_number"], f["date"], CABIN) for f in new_fl) * npax
        legs_txt = " then ".join(f"{f['flight_number']} on {f['date']}" for f in new_fl)
        tb = (f"Customer request (user {USER}): cancel reservation {R} completely, then book a NEW reservation for the same user and the "
              f"same passengers as currently on the reservation (copy their names and dates of birth exactly as they are now), "
              f"{res.flight_type.replace('_', ' ')} from {res.origin} to {res.destination} on exactly these flights in this order: "
              f"{legs_txt}; cabin {CABIN}; one free checked bag in total (no non-free bags); no travel insurance; pay the full price "
              f"with {CARD}. Look up each flight's {CABIN} price for its date and charge the exact total for all passengers.")

        def _book(st, be, R=R, USER=USER, res=res, CABIN=CABIN, new_fl=new_fl, CARD=CARD, total=total):
            cur = _cur(st, R)     # "the same passengers": as they are on the reservation at booking time (a partner may have fixed a name)
            pax_now = [{"first_name": p["first_name"], "last_name": p["last_name"], "dob": p["dob"]} for p in cur["passengers"]]
            return ("airline.book_reservation", {"user_id": USER, "origin": res.origin, "destination": res.destination, "flight_type": res.flight_type,
                                                 "cabin": CABIN, "flights": new_fl, "passengers": pax_now,
                                                 "payment_methods": [{"payment_id": CARD, "amount": total}], "total_baggages": 1, "nonfree_baggages": 0,
                                                 "insurance": "no"})
        ref_b = [("airline.cancel_reservation", {"reservation_id": R}), _book]
        p = pk[(i + 1) % 4]
        _add(serializable_cell(f"lt_cancel_rebook{sfx}", "tau2", Tau2AirlineBackend, tb, partner[p][0], ref_b, partner[p][1], ex, objs,
                               f"long: cancel+rebook vs {p}", SYSTEM_LONG))
    else:
        SKIPPED.append((f"lt_cancel_rebook{sfx}", ["no valid alternative itinerary"]))

    # ---- LT3: fix both passengers, add bags, change cabin (whole-list rewrites with a cabin-dependent allowance) --------
    tc = (f"Customer request for reservation {R}: (1) correct the passengers: {PAX[0][0]} {PAX[0][1]}'s date of birth is {new_dob}, and "
          f"{PAX[1][0]} {PAX[1][1]}'s last name is spelled '{new_last}'; keep everything else about the passengers unchanged. "
          f"(2) Then change the cabin of the whole reservation to {up_cabin} on the same flights and dates, charging any fare "
          f"difference to {CARD} (refund to it if cheaper). (3) Finally add two more checked bags — one free and one NON-free — "
          f"charging the fee to {CARD}: the total number of bags goes from {BAGS[0]} to {BAGS[0] + 2} and the non-free count from "
          f"{BAGS[1]} to {BAGS[1] + 1}.")
    ref_c = [_pax(R, 0, "dob", new_dob), _pax(R, 1, "last_name", new_last), _flights(R, up_cabin, {}, CARD), _bags(R, 2, 1, CARD)]
    p = pk[(i + 2) % 4]
    if p == "pax2":
        p = "leg2"     # the partner's passenger fix is part of this task already
    _add(serializable_cell(f"lt_pax_cabin_bags{sfx}", "tau2", Tau2AirlineBackend, tc, partner[p][0], ref_c, partner[p][1], ex, objs,
                           f"long: passengers+cabin+bags vs {p}", SYSTEM_LONG))
