"""Scripted (no-LLM) policy for the Micro cells: reads every listed item and customer in order, then writes by the rule, re-reading
and re-deriving the writes after a notification. Used by scripts/scripted_test.py and the unit tests."""
from __future__ import annotations

import json
import re
import time

from nexus.engine.scripted import PAUSE, clean
from benchmarks.micro.cells import CELLS


def _policy(cell, role):
    P = cell.params
    text = cell.task_a if role == "A" else cell.task_b
    w, d = P["w"], P["d"]
    items = [int(x) for x in re.findall(r"item (\d+)", text)] + [int(x) for x in re.findall(r"items ((?:\d+(?:, )?)+)", text) for x in []]
    all_items = sorted({int(x) for grp in re.findall(r"items? ((?:\d+(?:, )?)+)", text) for x in re.findall(r"\d+", grp)} | set(items))
    custs = sorted({int(x) for grp in re.findall(r"customers ((?:\d+(?:, )?)+)", text) for x in re.findall(r"\d+", grp)})
    writes = [int(x) for x in re.findall(r"for each of items ((?:\d+(?:, )?)+):", text)[0].split(",")] if re.findall(r"for each of items ((?:\d+(?:, )?)+):", text) else []
    thr = int(re.search(r"below (\d+)", text).group(1)); delta = int(re.search(r"plus (\d+)", text).group(1))

    def policy(messages):
        # positional history: a notification names the records whose values changed; reads (and writes) of those records that
        # precede the latest notification are stale and are redone. Everything after the notification stands.
        names, hist = {}, []
        note_idx, note = -1, ""
        for idx, m in enumerate(messages):
            if m["role"] == "assistant":
                for tc in m.get("tool_calls", []):
                    names[tc["id"]] = (tc["name"].replace("__", "."), tc["args"])
            elif m["role"] == "tool":
                n, a_ = names[m["tool_call_id"]]
                if n == "nexus.retract_my_actions":
                    hist = []
                    continue
                hist.append((idx, n, a_, m["content"]))
            if isinstance(m.get("content"), str) and "[NOTIFICATION]" in m["content"]:
                note_idx, note = idx, m["content"]
        stale = {i for i in all_items if f"stock:{w}.{i}" in note}
        reads, bals, done_w, done_c = {}, {}, set(), set()
        for idx, name, args, res in hist:
            if name == "tpcc.get_stock":
                i = int(args["item_id"])
                if idx <= note_idx and i in stale:
                    continue
                try:
                    reads[i] = json.loads(clean(res))["s_quantity"]
                except Exception:
                    pass
            elif name == "tpcc.get_customer":
                try:
                    bals[int(args["customer_id"])] = json.loads(clean(res))["c_balance"]
                except Exception:
                    pass
            elif name == "tpcc.set_stock":
                i = int(args["item_id"])
                if not (idx <= note_idx and i in stale) and not str(res).startswith("Action NOT executed"):
                    done_w.add(i)
            elif name == "tpcc.set_customer_balance":
                done_c.add(int(args["customer_id"]))
        for i in all_items:
            if i not in reads:
                return [("tpcc.get_stock", {"warehouse_id": w, "item_id": i})]
        for c in custs:
            if c not in bals:
                return [("tpcc.get_customer", {"warehouse_id": w, "district_id": d, "customer_id": c})]
        if not done_w and note_idx < 0:
            time.sleep(PAUSE)          # the window between reading and writing in which the partner's commit lands (as in the other suites)
        for i in writes:
            if i not in done_w and reads[i] < thr:
                return [("tpcc.set_stock", {"warehouse_id": w, "item_id": i, "quantity": reads[i] + delta})]
        for c in custs:
            if c not in done_c and bals.get(c, 0) < 0:
                return [("tpcc.set_customer_balance", {"warehouse_id": w, "district_id": d, "customer_id": c, "balance": 0})]
        return None
    return policy


POLICIES = {name: (lambda role, c=c: _policy(c, role)) for name, c in CELLS.items()}
