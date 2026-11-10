"""Ideal self-healing scripted policies for the TPC-C cell templates (parametrised by cells.PARAMS)."""
from __future__ import annotations

import json
import time

from nexus.engine.scripted import PAUSE, clean, note_text, notified, parse_json, prefix_record, tool_history
from benchmarks.tpcc.cells import PARAMS


def _last(h, name):
    xs = [x for x in h if x[0] == name]
    return xs[-1] if xs else None


def _stock_from_note(messages):
    rec = prefix_record(messages, "stock")
    return int(rec["s_quantity"]) if rec else None


def _order_then_done(w, d, c, item, qty):
    get = ("tpcc.get_stock", {"warehouse_id": w, "item_id": item})

    def place(q):
        return ("tpcc.place_order", {"warehouse_id": w, "district_id": d, "customer_id": c, "items": [{"item_id": item, "quantity": q}]})

    def policy(messages):
        h = tool_history(messages)
        if not h:
            return [get]
        placed = _last(h, "tpcc.place_order")
        if notified(messages):
            nt = note_text(messages)
            if placed and ("UNDONE" in nt or "rejected" in nt):
                q = qty(_stock_from_note(messages) or 0) if callable(qty) else qty
                return [get] if callable(qty) else [place(q)]
            if callable(qty) and placed:
                new_stock = _stock_from_note(messages)
                if new_stock is None:
                    return [get]
                oid = json.loads(clean(placed[2])).get("order_id") if clean(placed[2]).startswith("{") else None
                calls = [("tpcc.cancel_order", {"order_id": oid})] if oid else []
                calls.append(place(qty(new_stock)))
                return calls
            if not placed:
                return [get]
            return None
        last_tool, _, last_res = h[-1]
        if last_tool == get[0] and not placed:
            s = parse_json(last_res); q = qty(s["s_quantity"]) if callable(qty) else qty
            time.sleep(PAUSE)
            return [place(q)]
        if last_tool == get[0] and placed and callable(qty):
            s = parse_json(last_res); oid = json.loads(clean(placed[2])).get("order_id") if clean(placed[2]).startswith("{") else None
            calls = [("tpcc.cancel_order", {"order_id": oid})] if oid else []
            calls.append(place(qty(s["s_quantity"])))
            return calls
        if last_tool == get[0] and placed:
            return [place(qty)]           # the order was undone (MTPO): re-issue it
        if last_tool == "tpcc.place_order" and "Insufficient" in last_res:
            return [get]
        return None
    return policy


def two_neworders_hot_item(P, role):
    return _order_then_done(P["w"], P["d"], P["c1"], P["item"], P["q1"]) if role == "A" else _order_then_done(P["w"], P["d"], P["c2"], P["item"], P["q2"])


def neworder_vs_derived_restock(P, role):
    if role == "A":
        return _order_then_done(P["w"], P["d"], P["c"], P["item"], P["q"])
    get = ("tpcc.get_stock", {"warehouse_id": P["w"], "item_id": P["item"]})

    def policy(messages):
        h = tool_history(messages)
        if not h:
            return [get]
        if notified(messages):
            q = _stock_from_note(messages)
            if q is None:
                return [get]
            return [("tpcc.set_stock", {"warehouse_id": P["w"], "item_id": P["item"], "quantity": q + P["add"]})]
        last_tool, _, last_res = h[-1]
        if last_tool == get[0]:
            q = parse_json(last_res)["s_quantity"]
            if not _last(h, "tpcc.set_stock"):
                time.sleep(2 * PAUSE)
            return [("tpcc.set_stock", {"warehouse_id": P["w"], "item_id": P["item"], "quantity": q + P["add"]})]
        return None
    return policy


def balance_correction_vs_payment(P, role):
    w, d, c = P["w"], P["d"], P["c"]
    get = ("tpcc.get_customer", {"warehouse_id": w, "district_id": d, "customer_id": c})
    if role == "B":
        def policy(messages):
            h = tool_history(messages)
            if not h:
                return [get]
            if _last(h, "tpcc.record_payment"):
                if notified(messages) and ("rejected" in note_text(messages) or "UNDONE" in note_text(messages)):
                    return [("tpcc.record_payment", {"warehouse_id": w, "district_id": d, "customer_id": c, "amount": P["pay"]})]
                return None
            time.sleep(PAUSE)
            return [("tpcc.record_payment", {"warehouse_id": w, "district_id": d, "customer_id": c, "amount": P["pay"]})]
        return policy

    def policy(messages):
        h = tool_history(messages)
        if not h or notified(messages):
            return [get]
        last_tool, _, last_res = h[-1]
        if last_tool == get[0]:
            bal = parse_json(last_res)["c_balance"]
            if not _last(h, "tpcc.set_customer_balance"):
                time.sleep(2 * PAUSE)
            return [("tpcc.set_customer_balance", {"warehouse_id": w, "district_id": d, "customer_id": c, "balance": round(bal - P["delta"], 2)})]
        return None
    return policy


def deliver_vs_cancel(P, role):
    oid = P["oid"]
    get = ("tpcc.get_order", {"order_id": oid})
    act = ("tpcc.deliver_order", {"order_id": oid, "carrier_id": 4}) if role == "A" else ("tpcc.cancel_order", {"order_id": oid})

    def policy(messages):
        h = tool_history(messages)
        if not h or notified(messages):
            return [get]
        last_tool, _, last_res = h[-1]
        if last_tool == get[0]:
            o = parse_json(last_res)
            if o["o_status"] != "open" or _last(h, act[0]):
                return None
            time.sleep(PAUSE)
            return [act]
        return None
    return policy


def restock_then_half_order(P, role):
    if role == "A":
        get = ("tpcc.get_stock", {"warehouse_id": P["w"], "item_id": P["item"]})

        def policy(messages):
            h = tool_history(messages)
            if not h:
                return [get]
            if _last(h, "tpcc.set_stock"):
                return None
            time.sleep(PAUSE)
            return [("tpcc.set_stock", {"warehouse_id": P["w"], "item_id": P["item"], "quantity": P["set_q"]})]
        return policy
    return _order_then_done(P["w"], P["d"], P["c"], P["item"], lambda s: s // 2)


_T = {"two_neworders_hot_item": two_neworders_hot_item, "neworder_vs_derived_restock": neworder_vs_derived_restock,
      "balance_correction_vs_payment": balance_correction_vs_payment, "deliver_vs_cancel": deliver_vs_cancel,
      "restock_then_half_order": restock_then_half_order}
POLICIES = {name: (lambda role, P=P: _T[P["template"]](P, role)) for name, P in PARAMS.items()}
