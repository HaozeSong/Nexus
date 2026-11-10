"""Agentic TPC-C-lite on SQLite: a real relational store with structured, footprint-declared tools.

Schema (subset of TPC-C): warehouse, district, customer, item, stock, orders, order_line.  Tasks are the
TPC-C transactions phrased in natural language (NewOrder = place_order, Payment = record_payment,
Delivery = deliver_order, plus restock and cancel), executed by the agent through tools.  Hot spots are
controlled by drawing item ids from a Zipf distribution (see cells.py / workload).
"""
from __future__ import annotations

import json
import re
import random
import sqlite3

from nexus.backend import Backend
from nexus.tools import ToolSpec, schemas_from

N_W, N_D, N_C, N_I = 2, 3, 30, 100      # warehouses, districts/warehouse, customers/district, items
SEED = 7

DDL = """
CREATE TABLE warehouse(w_id INTEGER PRIMARY KEY, w_name TEXT, w_ytd REAL);
CREATE TABLE district(w_id INTEGER, d_id INTEGER, d_name TEXT, d_ytd REAL, d_next_o_id INTEGER, PRIMARY KEY(w_id,d_id));
CREATE TABLE customer(w_id INTEGER, d_id INTEGER, c_id INTEGER, c_name TEXT, c_balance REAL, c_ytd_payment REAL, c_payment_cnt INTEGER, PRIMARY KEY(w_id,d_id,c_id));
CREATE TABLE item(i_id INTEGER PRIMARY KEY, i_name TEXT, i_price REAL);
CREATE TABLE stock(w_id INTEGER, i_id INTEGER, s_quantity INTEGER, s_ytd INTEGER, s_order_cnt INTEGER, PRIMARY KEY(w_id,i_id));
CREATE TABLE orders(o_id INTEGER PRIMARY KEY, w_id INTEGER, d_id INTEGER, c_id INTEGER, o_entry_d TEXT, o_carrier_id INTEGER, o_ol_cnt INTEGER, o_total REAL, o_status TEXT);
CREATE TABLE order_line(o_id INTEGER, ol_number INTEGER, i_id INTEGER, ol_quantity INTEGER, ol_amount REAL, PRIMARY KEY(o_id,ol_number));
"""

NAMES = ["Ada", "Ben", "Chloe", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivan", "Jun", "Kai", "Lia", "Max", "Nia", "Omar",
         "Pia", "Quin", "Rey", "Sam", "Tia", "Uma", "Vic", "Wen", "Xan", "Yara", "Zed", "Ari", "Bea", "Cy", "Dee"]
ITEM_WORDS = ["bolt", "gear", "lamp", "valve", "cable", "sensor", "bracket", "pump", "filter", "motor"]


def build_pristine() -> sqlite3.Connection:
    rng = random.Random(SEED)
    con = sqlite3.connect(":memory:", check_same_thread=False)
    con.executescript(DDL)
    for w in range(1, N_W + 1):
        con.execute("INSERT INTO warehouse VALUES (?,?,?)", (w, f"Warehouse-{w}", 0.0))
        for d in range(1, N_D + 1):
            con.execute("INSERT INTO district VALUES (?,?,?,?,?)", (w, d, f"District-{w}-{d}", 0.0, 1))
            for c in range(1, N_C + 1):
                name = f"{NAMES[(c - 1) % len(NAMES)]} {chr(65 + (w * 7 + d * 3 + c) % 26)}."
                con.execute("INSERT INTO customer VALUES (?,?,?,?,?,?,?)", (w, d, c, name, round(rng.uniform(-50, 500), 2), 0.0, 0))
    for i in range(1, N_I + 1):
        con.execute("INSERT INTO item VALUES (?,?,?)", (i, f"{ITEM_WORDS[i % len(ITEM_WORDS)]}-{i:03d}", round(rng.uniform(5, 200), 2)))
        for w in range(1, N_W + 1):
            con.execute("INSERT INTO stock VALUES (?,?,?,?,?)", (w, i, rng.randint(10, 100), 0, 0))
    # a few pre-existing undelivered orders per district
    oid = 1
    for w in range(1, N_W + 1):
        for d in range(1, N_D + 1):
            for k in range(3):
                c = rng.randint(1, N_C)
                lines = [(rng.randint(1, N_I), rng.randint(1, 5)) for _ in range(rng.randint(1, 3))]
                total = 0.0
                for n, (i, q) in enumerate(lines, 1):
                    price = con.execute("SELECT i_price FROM item WHERE i_id=?", (i,)).fetchone()[0]
                    con.execute("INSERT INTO order_line VALUES (?,?,?,?,?)", (oid, n, i, q, round(price * q, 2)))
                    total += price * q
                con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", (oid, w, d, c, f"2024-06-0{k+1}", None, len(lines), round(total, 2), "open"))
                oid += 1
    con.execute("UPDATE district SET d_next_o_id=?", (oid,))
    con.commit()
    return con


PRISTINE = build_pristine()


def _clone(src: sqlite3.Connection) -> sqlite3.Connection:
    dst = sqlite3.connect(":memory:", check_same_thread=False)
    src.backup(dst)
    return dst


def _rows(con, sql, args=()):
    cur = con.execute(sql, args)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------- tools (each returns a JSON/str result)
def get_item(con, op, item_id: int):
    r = _rows(con, "SELECT * FROM item WHERE i_id=?", (item_id,))
    return json.dumps(r[0]) if r else "Item not found."


def search_items(con, op, name_contains: str = ""):
    return json.dumps(_rows(con, "SELECT * FROM item WHERE i_name LIKE ? LIMIT 20", (f"%{name_contains}%",)))


def get_stock(con, op, warehouse_id: int, item_id: int):
    r = _rows(con, "SELECT * FROM stock WHERE w_id=? AND i_id=?", (warehouse_id, item_id))
    return json.dumps(r[0]) if r else "Stock record not found."


def get_customer(con, op, warehouse_id: int, district_id: int, customer_id: int):
    r = _rows(con, "SELECT * FROM customer WHERE w_id=? AND d_id=? AND c_id=?", (warehouse_id, district_id, customer_id))
    return json.dumps(r[0]) if r else "Customer not found."


def find_customers(con, op, warehouse_id: int, district_id: int, name_contains: str = ""):
    return json.dumps(_rows(con, "SELECT * FROM customer WHERE w_id=? AND d_id=? AND c_name LIKE ? LIMIT 20",
                            (warehouse_id, district_id, f"%{name_contains}%")))


def get_order(con, op, order_id: int):
    o = _rows(con, "SELECT * FROM orders WHERE o_id=?", (order_id,))
    if not o:
        return "Order not found."
    o[0]["lines"] = _rows(con, "SELECT ol_number,i_id,ol_quantity,ol_amount FROM order_line WHERE o_id=? ORDER BY ol_number", (order_id,))
    return json.dumps(o[0])


def list_orders(con, op, warehouse_id: int, district_id: int, customer_id: int | None = None, status: str | None = None):
    sql, args = "SELECT o_id,c_id,o_entry_d,o_carrier_id,o_ol_cnt,o_total,o_status FROM orders WHERE w_id=? AND d_id=?", [warehouse_id, district_id]
    if customer_id is not None:
        sql += " AND c_id=?"; args.append(customer_id)
    if status:
        sql += " AND o_status=?"; args.append(status)
    return json.dumps(_rows(con, sql + " ORDER BY o_id LIMIT 50", args))


def place_order(con, op, warehouse_id: int, district_id: int, customer_id: int, items: list):
    """NewOrder: items = [{"item_id": int, "quantity": int}, ...]. Decrements stock (RMW), allocates the order id."""
    if not _rows(con, "SELECT 1 FROM customer WHERE w_id=? AND d_id=? AND c_id=?", (warehouse_id, district_id, customer_id)):
        return "Customer not found."
    if not items:
        return "No items given."
    total = 0.0
    plan = []
    for it in items:
        i, q = int(it["item_id"]), int(it["quantity"])
        row = _rows(con, "SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (warehouse_id, i))
        price = _rows(con, "SELECT i_price FROM item WHERE i_id=?", (i,))
        if not row or not price:
            return f"Item {i} not stocked in warehouse {warehouse_id}."
        if row[0]["s_quantity"] < q:
            return f"Insufficient stock for item {i}: {row[0]['s_quantity']} available, {q} requested. Order not placed."
        plan.append((i, q, price[0]["i_price"]))
    oid = int(op.next_id("orders"))
    for n, (i, q, price) in enumerate(plan, 1):
        con.execute("UPDATE stock SET s_quantity=s_quantity-?, s_ytd=s_ytd+?, s_order_cnt=s_order_cnt+1 WHERE w_id=? AND i_id=?", (q, q, warehouse_id, i))
        con.execute("INSERT INTO order_line VALUES (?,?,?,?,?)", (oid, n, i, q, round(price * q, 2)))
        total += price * q
    con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", (oid, warehouse_id, district_id, customer_id, "2024-06-15", None, len(plan), round(total, 2), "open"))
    con.execute("UPDATE district SET d_next_o_id=? WHERE w_id=? AND d_id=?", (oid + 1, warehouse_id, district_id))
    con.commit()
    return json.dumps({"order_id": oid, "total": round(total, 2), "status": "open"})


def set_stock(con, op, warehouse_id: int, item_id: int, quantity: int):
    """Restock: blind write of the on-hand quantity."""
    n = con.execute("UPDATE stock SET s_quantity=? WHERE w_id=? AND i_id=?", (quantity, warehouse_id, item_id)).rowcount
    con.commit()
    return "Stock updated." if n else "Stock record not found."


def record_payment(con, op, warehouse_id: int, district_id: int, customer_id: int, amount: float):
    """Payment: RMW on customer balance, district and warehouse year-to-date."""
    r = _rows(con, "SELECT c_balance FROM customer WHERE w_id=? AND d_id=? AND c_id=?", (warehouse_id, district_id, customer_id))
    if not r:
        return "Customer not found."
    con.execute("UPDATE customer SET c_balance=c_balance-?, c_ytd_payment=c_ytd_payment+?, c_payment_cnt=c_payment_cnt+1 WHERE w_id=? AND d_id=? AND c_id=?",
                (amount, amount, warehouse_id, district_id, customer_id))
    con.execute("UPDATE district SET d_ytd=d_ytd+? WHERE w_id=? AND d_id=?", (amount, warehouse_id, district_id))
    con.execute("UPDATE warehouse SET w_ytd=w_ytd+? WHERE w_id=?", (amount, warehouse_id))
    con.commit()
    nb = r[0]["c_balance"] - amount
    return json.dumps({"customer_id": customer_id, "new_balance": round(nb, 2)})


def set_customer_balance(con, op, warehouse_id: int, district_id: int, customer_id: int, balance: float):
    """Administrative correction: blind write of the customer's balance."""
    n = con.execute("UPDATE customer SET c_balance=? WHERE w_id=? AND d_id=? AND c_id=?", (balance, warehouse_id, district_id, customer_id)).rowcount
    con.commit()
    return "Balance updated." if n else "Customer not found."


def deliver_order(con, op, order_id: int, carrier_id: int):
    o = _rows(con, "SELECT o_status FROM orders WHERE o_id=?", (order_id,))
    if not o:
        return "Order not found."
    if o[0]["o_status"] != "open":
        return f"Order is {o[0]['o_status']}; only open orders can be delivered."
    con.execute("UPDATE orders SET o_carrier_id=?, o_status='delivered' WHERE o_id=?", (carrier_id, order_id))
    con.commit()
    return "Order delivered."


def cancel_order(con, op, order_id: int):
    o = _rows(con, "SELECT * FROM orders WHERE o_id=?", (order_id,))
    if not o:
        return "Order not found."
    if o[0]["o_status"] != "open":
        return f"Order is {o[0]['o_status']}; only open orders can be cancelled."
    lines = _rows(con, "SELECT i_id, ol_quantity FROM order_line WHERE o_id=?", (order_id,))
    for l in lines:
        con.execute("UPDATE stock SET s_quantity=s_quantity+? WHERE w_id=? AND i_id=?", (l["ol_quantity"], o[0]["w_id"], l["i_id"]))
    con.execute("UPDATE orders SET o_status='cancelled' WHERE o_id=?", (order_id,))
    con.commit()
    return (f"Order cancelled and stock restored in warehouse {o[0]['w_id']}: "
            + ", ".join(f"item {l['i_id']} +{l['ol_quantity']}" for l in lines) + ".")


TOOL_FUNCS = {"tpcc." + f.__name__: f for f in [get_item, search_items, get_stock, get_customer, find_customers, get_order,
                                                list_orders, place_order, set_stock, record_payment, set_customer_balance, deliver_order, cancel_order]}


def _stock_obj(k):
    return f"stock:{k['warehouse_id']}.{k['item_id']}"


def _cust_obj(k):
    return f"customer:{k['warehouse_id']}.{k['district_id']}.{k['customer_id']}"


def _orders_pred(k):
    """The predicate object of a district's order listing (`orders_<w>.<d>`): an insert into a district conflicts with listings of
    THAT district only (predicate scope). With the whole-table range `orders:*`, every delivery review in every district would
    repair at every bulk order of the run and exhaust its step budget at N=16."""
    return f"orders_{k['warehouse_id']}.{k['district_id']}"


def _fp_list_orders(kwargs, result, alloc):
    r = {_orders_pred(kwargs) + ":*"}
    if result:
        try:
            rows = json.loads(result)
            if isinstance(rows, list):
                r |= {f"orders:{row['o_id']}" for row in rows if isinstance(row, dict) and "o_id" in row}
        except (json.JSONDecodeError, TypeError):
            pass
    return r, set()


def _fp_place_order(kwargs, result, alloc):
    r = {_cust_obj(kwargs), f"district:{kwargs['warehouse_id']}.{kwargs['district_id']}"}
    w = {f"district:{kwargs['warehouse_id']}.{kwargs['district_id']}", _orders_pred(kwargs) + ":+"}   # insert: conflicts with listings of this district only
    for it in kwargs.get("items", []) or []:
        obj = f"stock:{kwargs['warehouse_id']}.{it.get('item_id')}"
        r.add(obj); w.add(obj); r.add(f"item:{it.get('item_id')}")
    w |= {f"orders:{i}" for i in alloc}
    return r, w


def _fp_payment(kwargs, result, alloc):
    objs = {_cust_obj(kwargs), f"district:{kwargs['warehouse_id']}.{kwargs['district_id']}", f"warehouse:{kwargs['warehouse_id']}"}
    return set(objs), set(objs)


def _fp_cancel(kwargs, result, alloc, lookup=None):
    """Cancel restores the order's lines to stock: the write set is those stock records exactly. After execution they are read off
    the result; before it (locks / intents / premise check) off the order's lines in the caller's view of the database. Only when
    neither is available does it fall back to the whole stock table (a table-level write invalidates every stock observation of
    every other task — )."""
    o = f"orders:{kwargs['order_id']}"
    if result is not None:
        m = re.search(r"restored in warehouse (\d+): (.*)\.$", result)
        if not m:
            return {o}, {o}          # refused (not open / not found): nothing restored
        return {o}, {o} | {f"stock:{m.group(1)}.{i}" for i in re.findall(r"item (\d+) \+", m.group(2))}
    if lookup is not None:
        try:
            rec, lines = lookup("orders", kwargs["order_id"]), lookup("order_lines", kwargs["order_id"])
        except Exception:
            rec, lines = None, None
        if rec is not None and lines is not None:
            return {o}, {o} | {f"stock:{rec['w_id']}.{l['i_id']}" for l in lines}
    return {o}, {o, "stock:*"}


SPECS = {s.name: s for s in [
    ToolSpec("tpcc.get_item", "read", "item", "item_id"),
    ToolSpec("tpcc.search_items", "read", "item", id_col="i_id"),
    ToolSpec("tpcc.get_stock", "read", "stock", footprint=lambda k, r, a: ({_stock_obj(k)}, set())),
    ToolSpec("tpcc.get_customer", "read", "customer", footprint=lambda k, r, a: ({_cust_obj(k)}, set())),
    ToolSpec("tpcc.find_customers", "read", "customer"),
    ToolSpec("tpcc.get_order", "read", "orders", "order_id"),
    ToolSpec("tpcc.list_orders", "read", "orders", footprint=_fp_list_orders),
    ToolSpec("tpcc.place_order", "write", "orders", footprint=_fp_place_order),
    ToolSpec("tpcc.set_stock", "write", "stock", footprint=lambda k, r, a: (set(), {_stock_obj(k)})),
    ToolSpec("tpcc.record_payment", "write", "customer", footprint=_fp_payment),
    ToolSpec("tpcc.set_customer_balance", "write", "customer", footprint=lambda k, r, a: (set(), {_cust_obj(k)})),
    ToolSpec("tpcc.deliver_order", "write", "orders", "order_id"),
    ToolSpec("tpcc.cancel_order", "write", "orders", footprint=_fp_cancel),
]}

_P = {"warehouse_id": {"type": "integer"}, "district_id": {"type": "integer"}, "customer_id": {"type": "integer"},
      "item_id": {"type": "integer"}, "order_id": {"type": "integer"}}
_DESC = {
    "tpcc.get_item": ("Get an item's name and price.", {"type": "object", "properties": {"item_id": _P["item_id"]}, "required": ["item_id"]}),
    "tpcc.search_items": ("Search items by name substring (max 20).", {"type": "object", "properties": {"name_contains": {"type": "string"}}}),
    "tpcc.get_stock": ("Get the stock record (on-hand quantity) of an item in a warehouse.", {"type": "object", "properties": {"warehouse_id": _P["warehouse_id"], "item_id": _P["item_id"]}, "required": ["warehouse_id", "item_id"]}),
    "tpcc.get_customer": ("Get a customer record (balance, payments).", {"type": "object", "properties": {"warehouse_id": _P["warehouse_id"], "district_id": _P["district_id"], "customer_id": _P["customer_id"]}, "required": ["warehouse_id", "district_id", "customer_id"]}),
    "tpcc.find_customers": ("Find customers of a district by name substring.", {"type": "object", "properties": {"warehouse_id": _P["warehouse_id"], "district_id": _P["district_id"], "name_contains": {"type": "string"}}, "required": ["warehouse_id", "district_id"]}),
    "tpcc.get_order": ("Get an order with its lines.", {"type": "object", "properties": {"order_id": _P["order_id"]}, "required": ["order_id"]}),
    "tpcc.list_orders": ("List orders of a district, optionally filtered by customer and status (open/delivered/cancelled).", {"type": "object", "properties": {"warehouse_id": _P["warehouse_id"], "district_id": _P["district_id"], "customer_id": _P["customer_id"], "status": {"type": "string"}}, "required": ["warehouse_id", "district_id"]}),
    "tpcc.place_order": ("Place a new order (TPC-C NewOrder): checks and decrements stock for each item, creates the order. Fails if stock is insufficient.", {"type": "object", "properties": {"warehouse_id": _P["warehouse_id"], "district_id": _P["district_id"], "customer_id": _P["customer_id"], "items": {"type": "array", "items": {"type": "object", "properties": {"item_id": _P["item_id"], "quantity": {"type": "integer"}}, "required": ["item_id", "quantity"]}}}, "required": ["warehouse_id", "district_id", "customer_id", "items"]}),
    "tpcc.set_stock": ("Set the on-hand quantity of an item in a warehouse (restock).", {"type": "object", "properties": {"warehouse_id": _P["warehouse_id"], "item_id": _P["item_id"], "quantity": {"type": "integer"}}, "required": ["warehouse_id", "item_id", "quantity"]}),
    "tpcc.record_payment": ("Record a customer payment (TPC-C Payment): reduces the customer's balance by amount and updates district/warehouse year-to-date totals.", {"type": "object", "properties": {"warehouse_id": _P["warehouse_id"], "district_id": _P["district_id"], "customer_id": _P["customer_id"], "amount": {"type": "number"}}, "required": ["warehouse_id", "district_id", "customer_id", "amount"]}),
    "tpcc.set_customer_balance": ("Administrative correction: set a customer's balance to an exact value.", {"type": "object", "properties": {"warehouse_id": _P["warehouse_id"], "district_id": _P["district_id"], "customer_id": _P["customer_id"], "balance": {"type": "number"}}, "required": ["warehouse_id", "district_id", "customer_id", "balance"]}),
    "tpcc.deliver_order": ("Mark an open order as delivered with a carrier id.", {"type": "object", "properties": {"order_id": _P["order_id"], "carrier_id": {"type": "integer"}}, "required": ["order_id", "carrier_id"]}),
    "tpcc.cancel_order": ("Cancel an open order and restore its items to stock.", {"type": "object", "properties": {"order_id": _P["order_id"]}, "required": ["order_id"]}),
}


class TPCCBackend(Backend):
    name = "tpcc"
    id_cols = {"orders": "o_id", "item": "i_id"}
    specs = SPECS

    def fresh_base(self):
        return _clone(PRISTINE)

    def copy(self, state):
        return _clone(state)

    def run_tool(self, state, tool, kwargs, op):
        return str(TOOL_FUNCS[tool](state, op, **kwargs))

    def initial_next_ids(self, base):
        return {"orders": int(base.execute("SELECT MAX(o_id) FROM orders").fetchone()[0]) + 1}

    def format_id(self, n):
        return str(n)

    def record(self, state, table, rid):
        if table == "orders":
            r = _rows(state, "SELECT * FROM orders WHERE o_id=?", (int(rid),))
        elif table == "stock":
            w, i = rid.split(".")
            r = _rows(state, "SELECT * FROM stock WHERE w_id=? AND i_id=?", (int(w), int(i)))
        elif table == "customer":
            w, d, c = rid.split(".")
            r = _rows(state, "SELECT * FROM customer WHERE w_id=? AND d_id=? AND c_id=?", (int(w), int(d), int(c)))
        elif table == "district":
            w, d = rid.split(".")
            r = _rows(state, "SELECT * FROM district WHERE w_id=? AND d_id=?", (int(w), int(d)))
        elif table == "warehouse":
            r = _rows(state, "SELECT * FROM warehouse WHERE w_id=?", (int(rid),))
        elif table == "order_lines":       # the lines of an order (footprint resolution of cancel_order), or None if no such order
            if not _rows(state, "SELECT o_id FROM orders WHERE o_id=?", (int(rid),)):
                return None
            return _rows(state, "SELECT i_id, ol_quantity FROM order_line WHERE o_id=?", (int(rid),))
        else:
            r = []
        return r[0] if r else None

    def tool_schemas(self, style, names=None):
        return schemas_from(_DESC, style, names)


def query(state, sql, args=()):
    return _rows(state, sql, args)
