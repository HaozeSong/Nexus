"""Micro: a microbenchmark over the TPC-C database that controls read/write-set sizes and the positions of reads and writes within a
task. Each task is an analyse-then-act procedure: it reads a set of records (stock levels, customer balances), applies a
stated rule, and writes a subset of them; a pair of tasks shares `overlap` records. The variable of interest is where in the task
the shared reads and the writes occur (`read_pos`, `write_pos` in [0,1]), i.e. how much reasoning a concurrent commit invalidates
and how much of it anchored replay can reuse. Correctness is the serializable outcome of the two reference procedures
(nexus.serial), like the other long-transaction benchmarks.

Configurations (name suffix): micro_r{R}_w{W}_p{P}_q{Q}_<k>: R records read, W records written, read position P and write position Q
in tenths. The default grid covers R in {2,4,8}, W in {1,2,4}, P and Q in {0.1, 0.5, 0.9}; `make_cells` builds any grid.
"""
from __future__ import annotations

from nexus.serial import serializable_cell
from benchmarks.tpcc.backend import N_D, N_I, N_W, TPCCBackend, query
from benchmarks.tpcc.cells import SYSTEM_EXTRA

CELLS: dict = {}
SKIPPED: list = []
SYSTEM_MICRO = SYSTEM_EXTRA + (" Work through the requested records one by one in the order given; read each record before you decide; "
                               "apply the rule exactly as stated and report what you changed.")


def _q(st, sql, args=()):
    return query(st, sql, args)


def _extract(w, items, custs):
    def extract(st):
        stock = tuple((i, _q(st, "SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, i))[0]["s_quantity"]) for i in items)
        cust = tuple((c, round(r["c_balance"], 2)) for (ww, d, c) in custs
                     for r in _q(st, "SELECT c_balance FROM customer WHERE w_id=? AND d_id=? AND c_id=?", (ww, d, c)))
        return (stock, cust)
    return extract


def _review(w, d, read_items, write_items, read_custs, threshold, delta):
    """Reference procedure: read the items and customers listed (reads come first, in the given order), then for every write item
    whose on-hand quantity is below `threshold` set it to the quantity read plus `delta` (a read-modify-write: two concurrent
    reviews that both top up the same item must compose, which is the lost-update anomaly the pair exercises), and for every
    customer read whose balance is negative set it to 0."""
    steps = []
    for i in read_items:
        steps.append(lambda st, be, i=i: None)        # a pure read: the reference does not change state for it
    for c in read_custs:
        steps.append(lambda st, be, c=c: None)
    for i in write_items:
        def f(st, be, i=i):
            q = _q(st, "SELECT s_quantity FROM stock WHERE w_id=? AND i_id=?", (w, i))[0]["s_quantity"]
            return ("tpcc.set_stock", {"warehouse_id": w, "item_id": i, "quantity": q + delta}) if q < threshold else None
        steps.append(f)
    for c in read_custs:
        def g(st, be, c=c):
            b = _q(st, "SELECT c_balance FROM customer WHERE w_id=? AND d_id=? AND c_id=?", (w, d, c))[0]["c_balance"]
            return ("tpcc.set_customer_balance", {"warehouse_id": w, "district_id": d, "customer_id": c, "balance": 0}) if b < 0 else None
        steps.append(g)
    return [s for s in steps if s is not None]


def _task_text(w, d, read_items, read_custs, write_items, threshold, delta, read_pos, write_pos):
    """The task text orders the work so that the shared reads land at `read_pos` and the writes at `write_pos` of the task: private
    bookkeeping reads (unshared records) pad the beginning and the middle."""
    parts = []
    pad = max(0, int(round(read_pos * 6)))
    for i in read_items[:pad]:
        parts.append(f"look up the on-hand quantity of item {i} in warehouse {w} (just note it)")
    parts.append("look up the on-hand quantity of items " + ", ".join(str(i) for i in read_items[pad:]) + f" in warehouse {w}")
    if read_custs:
        parts.append("look up the balance of customers " + ", ".join(str(c) for c in read_custs) + f" in warehouse {w}, district {d}")
    mid = max(0, int(round(write_pos * 6)) - pad)
    for i in read_items[pad:pad + mid]:
        parts.append(f"double-check item {i}'s record")
    parts.append(f"then, for each of items {', '.join(str(i) for i in write_items)}: if its on-hand quantity is below {threshold}, set its "
                 f"stock to the quantity you looked up plus {delta} (top it up by {delta}); leave the others unchanged")
    if read_custs:
        parts.append("and for each of the customers looked up: if the balance is negative, set it to 0 with the administrative balance correction")
    return f"Inventory and credit review for warehouse {w}, district {d}: " + "; ".join(parts) + ". Report what you changed."


def make_cells(grid=((2, 1), (4, 2), (8, 4)), positions=(0.1, 0.5, 0.9), n_inst: int = 2, threshold: int = 500, delta: int = 60):
    """Build Micro cells: for each (R, W), read position P, write position Q and instance k, a pair of tasks over disjoint private
    records plus `W` shared items that both tasks read and write."""
    out = {}
    next_item = [1]
    next_cust = [1]

    def take_items(n):
        xs = list(range(next_item[0], next_item[0] + n)); next_item[0] += n
        return [((x - 1) % N_I) + 1 for x in xs]

    def take_custs(n):
        xs = list(range(next_cust[0], next_cust[0] + n)); next_cust[0] += n
        return [((x - 1) % 30) + 1 for x in xs]
    k = 0
    for (R, W) in grid:
        for P in positions:
            for Q in positions:
                for inst in range(n_inst):
                    w = 1 + (k % N_W); d = 1 + (k % N_D)
                    shared = take_items(W)
                    pa = take_items(max(0, R - W)); pb = take_items(max(0, R - W))
                    ca = take_custs(max(1, R // 2)); cb = take_custs(max(1, R // 2))
                    ra = pa + shared; rb = pb + shared
                    name = f"micro_r{R}_w{W}_p{int(P * 10)}_q{int(Q * 10)}_{inst + 1}"
                    ta = _task_text(w, d, ra, ca, shared, threshold, delta, P, Q)
                    tb = _task_text(w, d, rb, cb, shared, threshold, delta, P, Q)
                    objs = {f"stock:{w}.{i}" for i in ra + rb} | {f"cust:{w}.{d}.{c}" for c in ca + cb}
                    ex = _extract(w, sorted(set(ra + rb)), [(w, d, c) for c in ca + cb])
                    cell = serializable_cell(name, "micro", TPCCBackend, ta, tb, _review(w, d, ra, shared, ca, threshold, delta),
                                             _review(w, d, rb, shared, cb, threshold, delta), ex, objs,
                                             f"micro: R={R} W={W} read_pos={P} write_pos={Q}", SYSTEM_MICRO)
                    cell.params = {"R": R, "W": W, "read_pos": P, "write_pos": Q, "shared": shared, "w": w, "d": d}
                    try:
                        if any(e for _, _, e in cell.expected()):
                            SKIPPED.append(name); continue
                    except Exception as e:
                        SKIPPED.append((name, str(e)[:80])); continue
                    out[name] = cell
                    k += 1
    return out


CELLS.update(make_cells())
