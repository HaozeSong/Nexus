"""Ideal self-healing scripted policies for the tau2 airline cell templates (parametrised by cells.PARAMS)."""
from __future__ import annotations

import time

from nexus.engine.scripted import PAUSE, note_text, notified, parse_json, prefix_record, tool_history
from benchmarks.tau2.cells import PARAMS


def _last(h, name):
    xs = [x for x in h if x[0] == name]
    return xs[-1] if xs else None


def _rewrite_policy(P, write_tool, make_kwargs, pause=PAUSE):
    R = P["R"]; GET = ("airline.get_reservation_details", {"reservation_id": R})

    def policy(messages):
        h = tool_history(messages)
        if not h:
            return [GET]
        if notified(messages):
            rec = prefix_record(messages, "reservations", R)
            if rec is None:
                return [GET]
            if rec.get("status") == "cancelled":
                return [("nexus.retract_my_actions", {"count": 1})] if _last(h, write_tool) and "UNDONE" not in note_text(messages) else None
            return [(write_tool, make_kwargs(rec))]
        last_tool, _, last_res = h[-1]
        if last_tool == GET[0]:
            rec = parse_json(last_res)
            if rec.get("status") == "cancelled":
                return None
            if not _last(h, write_tool):
                time.sleep(pause)
            return [(write_tool, make_kwargs(rec))]
        return None
    return policy


def passengers_lost_update(P, role):
    def mk(rec):
        pax = [dict(p) for p in rec["passengers"]]
        if role == "A":
            pax[0]["dob"] = P["new_dob"]
        else:
            pax[1]["last_name"] = P["new_last"]
        return {"reservation_id": P["R"], "passengers": pax}
    return _rewrite_policy(P, "airline.update_reservation_passengers", mk)


def baggage_lost_update(P, role):
    def mk(rec):
        t, n = rec["total_baggages"], rec["nonfree_baggages"]
        return ({"reservation_id": P["R"], "total_baggages": t + 1, "nonfree_baggages": n, "payment_id": P["CARD"]} if role == "A"
                else {"reservation_id": P["R"], "total_baggages": t + 2, "nonfree_baggages": n + 2, "payment_id": P["CARD"]})
    return _rewrite_policy(P, "airline.update_reservation_baggages", mk)


def flights_lost_update(P, role):
    def mk(rec):
        fl = [{"flight_number": f["flight_number"], "date": f["date"]} for f in rec["flights"]]
        i = 0 if role == "A" else 1
        fl[i] = {"flight_number": P["ALT"][i], "date": P["FLIGHTS"][i][1]}
        return {"reservation_id": P["R"], "cabin": rec["cabin"], "flights": fl, "payment_id": P["CARD"]}
    return _rewrite_policy(P, "airline.update_reservation_flights", mk)


def _bags_b(P):
    def mk(rec):
        return {"reservation_id": P["R"], "total_baggages": rec["total_baggages"] + 2, "nonfree_baggages": rec["nonfree_baggages"] + 2, "payment_id": P["CARD"]}
    return _rewrite_policy(P, "airline.update_reservation_baggages", mk, pause=2 * PAUSE)


def _cancel_a(P):
    R = P["R"]; GET = ("airline.get_reservation_details", {"reservation_id": R})

    def policy(messages):
        h = tool_history(messages)
        if not h:
            return [GET]
        if _last(h, "airline.cancel_reservation"):
            return None
        time.sleep(PAUSE)
        return [("airline.cancel_reservation", {"reservation_id": R})]
    return policy


def cancel_vs_bags(P, role):
    return _cancel_a(P) if role == "A" else _bags_b(P)


def cancel_certificate_vs_bags(P, role):
    if role == "B":
        return _bags_b(P)
    R, USER = P["R"], P["USER"]; GET = ("airline.get_reservation_details", {"reservation_id": R})

    def policy(messages):
        h = tool_history(messages)
        if not h:
            return [GET]
        if notified(messages):
            rec = prefix_record(messages, "reservations", R)
            if rec is None and _last(h, "airline.send_certificate"):
                return [GET]
            if rec and _last(h, "airline.send_certificate"):
                paid_before_cancel = sum(p["amount"] for p in rec["payment_history"] if p["amount"] > 0)
                sent = _last(h, "airline.send_certificate")[1]["amount"]
                if abs(paid_before_cancel - sent) > 0.5:
                    return [("nexus.retract_my_actions", {"count": 1}), ("airline.send_certificate", {"user_id": USER, "amount": int(paid_before_cancel)})]
            return None
        last_tool, _, last_res = h[-1]
        if last_tool == GET[0]:
            rec = parse_json(last_res)
            if rec.get("status") == "cancelled":
                return None
            paid = sum(p["amount"] for p in rec["payment_history"])
            time.sleep(PAUSE)
            return [("airline.cancel_reservation", {"reservation_id": R}), ("airline.send_certificate", {"user_id": USER, "amount": int(paid)})]
        return None
    return policy


_T = {"passengers_lost_update": passengers_lost_update, "baggage_lost_update": baggage_lost_update, "flights_lost_update": flights_lost_update,
      "cancel_vs_bags": cancel_vs_bags, "cancel_certificate_vs_bags": cancel_certificate_vs_bags}
POLICIES = {name: (lambda role, P=P: _T[P["template"]](P, role)) for name, P in PARAMS.items()}
