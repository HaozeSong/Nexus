"""Derived long transactions on the TPC-C-lite database (secondary long-transaction benchmark, 2026-09-23).

Unlike TPC-C's 3–5-step transactions, each task here is an analyse-then-act procedure over several rows: it reads a set of
records, applies a stated rule to each, and writes the rows that need changing (6–15 tool calls, every write derived from a
read). Pairs conflict on the same items / customers / district. Correctness = the final state equals one of the two serial
executions of the reference procedures (nexus.serial), with ids of new orders ignored."""
from __future__ import annotations

from nexus.serial import serializable_cell
from benchmarks.tpcc.backend import N_C, N_D, N_I, N_W, TPCCBackend, query
from benchmarks.tpcc.cells import SYSTEM_EXTRA

CELLS: dict = {}
SKIPPED: list = []
SYSTEM_LONG = SYSTEM_EXTRA + (" A request may cover several records: handle them one by one in the order given, reading each record "
                              "before deciding, and apply the rule exactly as stated (no rounding other than what is stated).")
ITEMS_PER = 6
_RANGES = [list(range(1 + k * ITEMS_PER, 1 + (k + 1) * ITEMS_PER)) for k in range(N_I // ITEMS_PER)]   # 16 disjoint item ranges
_DISTRICTS = [(w, d) for w in range(1, N_W + 1) for d in range(1, N_D + 1)]                          # 6 districts


def _q(st, sql, args=()):
    return query(st, sql, args)


def _extract_factory(w, items, custs, district):
    """items: list of item ids (stock rows of warehouse w); custs: [(w,d,c)]; district: (w,d) whose orders of those customers count."""
    def extract(st):
        stock = tuple((i, _q(st, "SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, i))[0]["s_quantity"]) for i in items)
        cust = tuple((c, round(r["c_balance"], 2), round(r["c_ytd_payment"], 2), r["c_payment_cnt"])
                     for (ww, d, c) in custs for r in _q(st, "SELECT * FROM customer WHERE w_id=? AND d_id=? AND c_id=?", (ww, d, c)))
        orders = []
        if district:
            dw, dd = district
            for o in _q(st, "SELECT o_id, c_id, o_status, o_carrier_id FROM orders WHERE w_id=? AND d_id=? ORDER BY o_id", (dw, dd)):
                if custs and (dw, dd, o["c_id"]) not in custs:
                    continue
                lines = tuple(sorted((l["i_id"], l["ol_quantity"]) for l in _q(st, "SELECT i_id, ol_quantity FROM order_line WHERE o_id=?", (o["o_id"],))))
                orders.append((o["c_id"], o["o_status"], o["o_carrier_id"], lines))
        return (stock, cust, tuple(sorted(orders)))
    return extract


# ---- reference procedures (state-dependent) ------------------------------------------------------------------------
def restock_sweep(w, items, low=60, target=120):
    """For each item: if on-hand < low, set it to target."""
    def steps():
        for i in items:
            def f(st, be, i=i):
                q = _q(st, "SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, i))[0]["s_quantity"]
                return ("tpcc.set_stock", {"warehouse_id": w, "item_id": i, "quantity": target}) if q < low else None
            yield f
    return list(steps())


def bulk_order(w, d, c, items, div=4, min_stock=8):
    """One order per item with quantity floor(on-hand/div); skip items with on-hand < min_stock."""
    def steps():
        for i in items:
            def f(st, be, i=i):
                q = _q(st, "SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, i))[0]["s_quantity"]
                return ("tpcc.place_order", {"warehouse_id": w, "district_id": d, "customer_id": c, "items": [{"item_id": i, "quantity": q // div}]}) if q >= min_stock else None
            yield f
    return list(steps())


def delivery_review(w, d, carrier=3):
    """For each open order of the district (ascending id): deliver it if the customer's balance is >= 0, otherwise cancel it.
    Every step handles the FIRST still-open order (a delivered/cancelled order leaves the open list), so no bookkeeping is needed."""
    def steps():
        for _ in range(12):
            def f(st, be):
                opens = _q(st, "SELECT o_id, c_id FROM orders WHERE w_id=? AND d_id=? AND o_status='open' ORDER BY o_id", (w, d))
                if not opens:
                    return None
                o = opens[0]
                bal = _q(st, "SELECT c_balance FROM customer WHERE w_id=? AND d_id=? AND c_id=?", (w, d, o["c_id"]))[0]["c_balance"]
                return ("tpcc.deliver_order", {"order_id": o["o_id"], "carrier_id": carrier}) if bal >= 0 else ("tpcc.cancel_order", {"order_id": o["o_id"]})
            yield f
    return list(steps())


def credit_review(w, d, custs, high=200, floor_to=100):
    """For each customer: balance > high -> record a payment of (balance - floor_to); balance < 0 -> set the balance to 0."""
    def steps():
        for c in custs:
            def f(st, be, c=c):
                bal = _q(st, "SELECT c_balance FROM customer WHERE w_id=? AND d_id=? AND c_id=?", (w, d, c))[0]["c_balance"]
                if bal > high:
                    return ("tpcc.record_payment", {"warehouse_id": w, "district_id": d, "customer_id": c, "amount": round(bal - floor_to, 2)})
                if bal < 0:
                    return ("tpcc.set_customer_balance", {"warehouse_id": w, "district_id": d, "customer_id": c, "balance": 0})
                return None
            yield f
    return list(steps())


def _add(cell):
    exp = cell.expected()
    if any(e for _, _, e in exp):
        SKIPPED.append((cell.name, [e for _, _, e in exp if e]))
        return
    CELLS[cell.name] = cell


def _items_txt(items):
    return ", ".join(str(i) for i in items)


# ---- pairs --------------------------------------------------------------------------------------------------------------
for k, items in enumerate(_RANGES):
    w = 1 + (k % N_W)
    d = 1 + (k % N_D)
    c = 1 + k                       # customers 1..16 of district (w,d): distinct per range within a district
    sfx = "" if k == 0 else f"_{k + 1}"
    objs = {f"stock:{w}.{i}" for i in items} | {f"cust:{w}.{d}.{c}"}     # district shared with other bulk orders (order-id counter = hot spot)
    ex = _extract_factory(w, items, [(w, d, c)], (w, d))
    ta = (f"Inventory review for warehouse {w}, items {_items_txt(items)}: for EACH of these items, look up the on-hand quantity in "
          f"warehouse {w}; if it is below 60, set it to 120 (leave items at or above 60 unchanged). Report what you changed.")
    tb = (f"Bulk order for customer {c} in warehouse {w}, district {d}, items {_items_txt(items)}: for EACH item, look up the current "
          f"on-hand quantity in warehouse {w}; if it is at least 8, place a separate order for that item alone with quantity equal to "
          f"the on-hand quantity divided by 4, rounded down; skip items with fewer than 8 on hand. Report the orders placed.")
    _add(serializable_cell(f"dl_restock_vs_bulk{sfx}", "tpcc", TPCCBackend, ta, tb, restock_sweep(w, items), bulk_order(w, d, c, items), ex, objs,
                           "derived long: restock sweep vs bulk order on the same items (lost update on stock, derived quantities)", SYSTEM_LONG))

for k, (w, d) in enumerate(_DISTRICTS):
    sfx = "" if k == 0 else f"_{k + 1}"
    opens = _q(TPCCBackend().fresh_base(), "SELECT o_id, c_id FROM orders WHERE w_id=? AND d_id=? AND o_status='open' ORDER BY o_id", (w, d))
    custs = sorted({o["c_id"] for o in opens})
    objs = {f"dorders:{w}.{d}"} | {f"cust:{w}.{d}.{c}" for c in range(1, N_C + 1)}   # reviews every open order: owns the district
    ex = _extract_factory(w, [], [(w, d, c) for c in custs], (w, d))
    ta = (f"Delivery review for warehouse {w}, district {d}: list the OPEN orders of this district and process them in ascending order id. "
          f"For each open order, look up the customer's current balance: if the balance is 0 or more, deliver the order with carrier 3; "
          f"if the balance is negative, cancel the order instead. Report each decision.")
    tb = (f"Credit review for customers {', '.join(str(c) for c in custs)} in warehouse {w}, district {d}: for EACH of these customers, look up "
          f"the current balance; if it is above 200, record a payment of (balance - 100) so that the balance becomes exactly 100; if it is "
          f"negative, set the balance to 0 with the administrative balance correction; otherwise leave the customer unchanged. Report what you did.")
    _add(serializable_cell(f"dl_delivery_vs_credit{sfx}", "tpcc", TPCCBackend, ta, tb, delivery_review(w, d), credit_review(w, d, custs), ex, objs,
                           "derived long: delivery decisions depend on balances another agent is correcting", SYSTEM_LONG))
