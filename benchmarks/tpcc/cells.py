"""Contended TPC-C-lite cells generated from five templates, each instantiated on N_INST disjoint record sets
(deterministic seed).  The first instance of a template carries the template's name; further instances add _2, _3, ..."""
from __future__ import annotations

import random

from nexus.cell import Cell
from benchmarks.tpcc.backend import N_C, N_D, N_I, N_W, PRISTINE, TPCCBackend, query

SYSTEM_EXTRA = ("You operate the company's order-management database (TPC-C style: warehouses, districts, "
                "customers, items, stock, orders). Use the tpcc tools; identifiers are integers. Always look up "
                "current values before computing a change.")
N_INST = 18
CELLS: dict[str, Cell] = {}
PARAMS: dict[str, dict] = {}


def _base(sql, args=()):
    return query(PRISTINE, sql, args)


def _live(store, sql, args=()):
    return query(store.materialize(), sql, args)


def _add(name, template, cell, params):
    cell.backend = TPCCBackend; cell.benchmark = "tpcc"; cell.system_extra = SYSTEM_EXTRA
    cell.objects = {f"tpcc:{name}"}       # every cell is generated on its own customers/items/orders
    cell.template = template
    CELLS[name] = cell; PARAMS[name] = dict(params, template=template)


rng = random.Random(11)
_items = list(range(1, N_I + 1)); rng.shuffle(_items)
_used_items = set()
_open_orders = [r["o_id"] for r in _base("SELECT o_id FROM orders WHERE o_status='open' ORDER BY o_id")]


def _take_item(preferred=None):
    if preferred and preferred not in _used_items:
        _used_items.add(preferred); return preferred
    for i in _items:
        if i not in _used_items:
            _used_items.add(i); return i
    raise RuntimeError("out of items")


def _name(base, i):
    return base if i == 0 else f"{base}_{i+1}"


# ---- T1 benign: two in-tool RMW NewOrders on one hot item ------------------------------------------
def t1(name, w, d, c1, q1, c2, q2, item):
    s0 = _base("SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, item))[0]["s_quantity"]
    price = _base("SELECT i_price FROM item WHERE i_id=?", (item,))[0]["i_price"]

    def inv(store):
        st = _live(store, "SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, item))[0]
        orders = _live(store, "SELECT o.o_id, o.c_id, o.o_total, l.ol_quantity FROM orders o JOIN order_line l ON o.o_id=l.o_id "
                              "WHERE o.w_id=? AND o.d_id=? AND l.i_id=? AND o.o_id>? AND o.o_status!='cancelled'", (w, d, item, len(_open_orders)))
        qa = [o for o in orders if o["c_id"] == c1 and o["ol_quantity"] == q1]; qb = [o for o in orders if o["c_id"] == c2 and o["ol_quantity"] == q2]
        if len(qa) != 1 or len(qb) != 1:
            return False, f"expected one order each (got A:{len(qa)}, B:{len(qb)})"
        if st["s_quantity"] != s0 - q1 - q2:
            return False, f"stock {st['s_quantity']} != {s0}-{q1}-{q2} (lost update)"
        for o, q in ((qa[0], q1), (qb[0], q2)):
            if abs(o["o_total"] - round(price * q, 2)) > 0.01:
                return False, f"order total mismatch {o}"
        return True, "both orders placed, stock consistent"
    _add(name, "two_neworders_hot_item", Cell(name, "benign footprint conflict: two in-tool RMWs on one hot stock row",
        task_a=f"Place a new order in warehouse {w}, district {d} for customer {c1}: {q1} units of item {item}.",
        task_b=f"Place a new order in warehouse {w}, district {d} for customer {c2}: {q2} units of item {item}.",
        invariant=inv), dict(w=w, d=d, c1=c1, q1=q1, c2=c2, q2=q2, item=item, s0=s0))


# ---- T2 lost update: in-tool RMW vs read-then-blind restock --------------------------------------------
def t2(name, w, d, c, item, q, add):
    s0 = _base("SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, item))[0]["s_quantity"]

    def inv(store):
        st = _live(store, "SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, item))[0]["s_quantity"]
        orders = _live(store, "SELECT o.o_id FROM orders o JOIN order_line l ON o.o_id=l.o_id WHERE o.w_id=? AND o.d_id=? AND o.c_id=? AND l.i_id=? AND l.ol_quantity=? AND o.o_id>? AND o.o_status!='cancelled'", (w, d, c, item, q, len(_open_orders)))
        if len(orders) != 1:
            return False, f"expected exactly one order, got {len(orders)}"
        if st == s0 + add - q:
            return True, f"stock {st} = {s0}+{add}-{q} (either order)"
        return False, f"stock {st} != {s0 + add - q} (lost update)"
    _add(name, "neworder_vs_derived_restock", Cell(name, "lost update: in-tool RMW decrement vs read-then-blind-set restock",
        task_a=f"Place a new order in warehouse {w}, district {d} for customer {c}: {q} units of item {item}.",
        task_b=f"A delivery of {add} units of item {item} arrived at warehouse {w}: look up its current on-hand quantity and set the stock to the current quantity plus {add}.",
        invariant=inv), dict(w=w, d=d, c=c, item=item, q=q, add=add, s0=s0))


# ---- T3 lost update: read-then-blind balance correction vs in-tool RMW payment -------------------------
def t3(name, w, d, c, delta, pay):
    b0 = _base("SELECT c_balance FROM customer WHERE w_id=? AND d_id=? AND c_id=?", (w, d, c))[0]["c_balance"]
    d0 = _base("SELECT d_ytd FROM district WHERE w_id=? AND d_id=?", (w, d))[0]["d_ytd"]

    def inv(store):
        cu = _live(store, "SELECT c_balance, c_ytd_payment, c_payment_cnt FROM customer WHERE w_id=? AND d_id=? AND c_id=?", (w, d, c))[0]
        dy = _live(store, "SELECT d_ytd FROM district WHERE w_id=? AND d_id=?", (w, d))[0]["d_ytd"]
        ok = abs(cu["c_balance"] - (b0 - delta - pay)) < 0.01 and abs(cu["c_ytd_payment"] - pay) < 0.01 and cu["c_payment_cnt"] == 1 and abs(dy - d0 - pay) < 0.01
        return (True, "correction and payment both reflected") if ok else (False, f"customer={cu} d_ytd={dy} (base balance {b0})")
    _add(name, "balance_correction_vs_payment", Cell(name, "lost update: read-then-blind-write correction vs in-tool RMW payment",
        task_a=f"Customer {c} of warehouse {w}, district {d} was overcharged by ${delta}: look up their current balance and correct it by setting the balance to the current value minus {delta}.",
        task_b=f"Record a ${pay} payment from customer {c} in warehouse {w}, district {d}.",
        invariant=inv), dict(w=w, d=d, c=c, delta=delta, pay=pay, b0=b0))


# ---- T4 benign: conflicting state transitions guarded inside the tools ----------------------------------
def t4(name, oid):
    o = _base("SELECT w_id, d_id FROM orders WHERE o_id=?", (oid,))[0]; w, d = o["w_id"], o["d_id"]
    lines = _base("SELECT i_id, ol_quantity FROM order_line WHERE o_id=?", (oid,))
    s0 = {l["i_id"]: _base("SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, l["i_id"]))[0]["s_quantity"] for l in lines}

    def inv(store):
        r = _live(store, "SELECT o_status, o_carrier_id FROM orders WHERE o_id=?", (oid,))[0]
        stocks = {l["i_id"]: _live(store, "SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, l["i_id"]))[0]["s_quantity"] for l in lines}
        if r["o_status"] == "delivered" and r["o_carrier_id"] == 4 and stocks == s0:
            return True, "A·B: delivered first, cancel refused"
        restored = {l["i_id"]: s0[l["i_id"]] + l["ol_quantity"] for l in lines}
        if r["o_status"] == "cancelled" and r["o_carrier_id"] is None and stocks == restored:
            return True, "B·A: cancelled first, delivery refused"
        return False, f"order={r} stocks={stocks} (base {s0})"
    _add(name, "deliver_vs_cancel", Cell(name, "benign: conflicting state transitions guarded inside the tools",
        task_a=f"Order {oid} (warehouse {w}, district {d}) is going out: mark it delivered with carrier 4.",
        task_b=f"The customer called to cancel order {oid} in warehouse {w}, district {d}. Cancel it so the stock is restored.",
        invariant=inv), dict(oid=oid, w=w, d=d))


# ---- T5 canary: order quantity derived from a value another agent rewrites ---------------------------------
def t5(name, w, d, c, item, set_q):
    s0 = _base("SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, item))[0]["s_quantity"]

    def inv(store):
        st = _live(store, "SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, item))[0]["s_quantity"]
        orders = _live(store, "SELECT o.o_id, l.ol_quantity FROM orders o JOIN order_line l ON o.o_id=l.o_id WHERE o.w_id=? AND o.d_id=? AND o.c_id=? AND l.i_id=? AND o.o_id>? AND o.o_status!='cancelled'", (w, d, c, item, len(_open_orders)))
        if len(orders) != 1:
            return False, f"expected one order, got {len(orders)}"
        q = orders[0]["ol_quantity"]
        if q == set_q // 2 and st == set_q - set_q // 2:
            return True, f"A·B: restock {set_q}, order {q}"
        if q == s0 // 2 and st == set_q:
            return True, f"B·A: order {q} from {s0}, then restock {set_q}"
        return False, f"order qty {q}, stock {st} (base {s0}, restock {set_q})"
    _add(name, "restock_then_half_order", Cell(name, "stale premise (canary): order quantity derived from a value another agent rewrites",
        task_a=f"Restock item {item} in warehouse {w}: set its on-hand quantity to {set_q}.",
        task_b=f"Place a new order in warehouse {w}, district {d} for customer {c} for item {item}, ordering exactly half (rounded down) of the item's current on-hand quantity in warehouse {w}.",
        invariant=inv), dict(w=w, d=d, c=c, item=item, set_q=set_q, s0=s0))


# ---- instantiate: the first instance of each template, then N_INST - 1 more ------------------------------------
t1("two_neworders_hot_item", 1, 1, 3, 6, 7, 9, _take_item(5))
t2("neworder_vs_derived_restock", 1, 2, 11, _take_item(12), 7, 100)
t3("balance_correction_vs_payment", 2, 1, 5, 120, 45)
t4("deliver_vs_cancel", _open_orders[6] if len(_open_orders) > 6 else _open_orders[0])
t5("restock_then_half_order", 2, 2, 2, _take_item(20), 60)
_used_orders = {PARAMS["deliver_vs_cancel"]["oid"]}
_used_cust = {(1, 1, 3), (1, 1, 7), (1, 2, 11), (2, 1, 5), (2, 2, 2)}


def _cust():
    while True:
        k = (rng.randint(1, N_W), rng.randint(1, N_D), rng.randint(1, N_C))
        if k not in _used_cust:
            _used_cust.add(k); return k


for i in range(1, N_INST):
    w, d, c1 = _cust(); _, _, c2 = _cust()
    t1(_name("two_neworders_hot_item", i), w, d, c1, rng.randint(2, 8), c2, rng.randint(2, 8), _take_item())
    w, d, c = _cust(); t2(_name("neworder_vs_derived_restock", i), w, d, c, _take_item(), rng.randint(3, 9), rng.choice([50, 80, 100, 120]))
    w, d, c = _cust(); t3(_name("balance_correction_vs_payment", i), w, d, c, rng.choice([60, 80, 100, 120, 150]), rng.choice([25, 45, 70, 90]))
    oid = next(o for o in _open_orders if o not in _used_orders); _used_orders.add(oid); t4(_name("deliver_vs_cancel", i), oid)
    w, d, c = _cust(); t5(_name("restock_then_half_order", i), w, d, c, _take_item(), rng.choice([40, 60, 80, 100]))
