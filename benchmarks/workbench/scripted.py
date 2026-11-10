"""Ideal self-healing scripted policies for the WorkBench cell templates (parametrised by cells.PARAMS)."""
from __future__ import annotations

import json
import time

from nexus.engine.scripted import PAUSE, clean, notified, parse_json, prefix_record, shift, tool_history
from benchmarks.workbench.cells import PARAMS


def _last(h, name):
    xs = [x for x in h if x[0] == name]
    return xs[-1] if xs else None


def crm_write_skew(P, role):
    search = ("customer_relationship_manager.search_customers",
              {"assigned_to_email": P["assignee"], "status": P["status"], "product_interest": P["interest"]})
    field, value, back = (("status", "Won", P["status"]) if role == "A" else ("assigned_to_email", P["new_assignee"], P["assignee"]))

    def policy(messages):
        h = tool_history(messages)
        if notified(messages) or not h:
            return [search]
        last_tool, _, last_res = h[-1]
        if last_tool == search[0]:
            ids = [r["customer_id"] for r in (parse_json(last_res) or [])]
            if len([x for x in h if x[0] == search[0]]) > 1:
                mine = {a["customer_id"] for t, a, _ in h if t.endswith("update_customer") and a.get("field") == field}
                gone = sorted(mine - set(ids))
                if gone:
                    return [("customer_relationship_manager.update_customer", {"customer_id": g, "field": field, "new_value": back}) for g in gone]
            if not ids:
                return None
            time.sleep(PAUSE)
            return [("customer_relationship_manager.update_customer", {"customer_id": i, "field": field, "new_value": value}) for i in ids]
        return None
    return policy


def calendar_first_meeting(P, role):
    d = P["date"]
    search = ("calendar.search_events", {"query": "", "time_min": f"{d} 00:00:00", "time_max": f"{d} 23:59:59"})

    def act(eid):
        return ("calendar.delete_event", {"event_id": eid}) if role == "A" else ("calendar.update_event", {"event_id": eid, "field": "event_start", "new_value": f"{d} 16:00:00"})

    def policy(messages):
        h = tool_history(messages)
        if notified(messages) or not h:
            return [search]
        last_tool, _, last_res = h[-1]
        if last_tool == search[0]:
            rows = sorted(parse_json(last_res) or [], key=lambda r: r["event_start"])
            rows = [r for r in rows if r["event_start"] != f"{d} 16:00:00"]
            if not rows:
                return None
            first = rows[0]["event_id"]
            done = {a["event_id"] for t, a, _ in h if t in ("calendar.delete_event", "calendar.update_event")}
            if first in done:
                return None
            if len([x for x in h if x[0] == search[0]]) == 1:
                time.sleep(PAUSE)
            return [act(first)]
        return None
    return policy


def pm_canary(P, role):
    if role == "A":
        search = ("project_management.search_tasks", {"assigned_to_email": P["assignee"], "list_name": P["lst"], "due_date": P["due"]})

        def policy(messages):
            h = tool_history(messages)
            if not h or notified(messages):
                return [search]
            if h[-1][0] == search[0]:
                ids = [r["task_id"] for r in (parse_json(h[-1][2]) or []) if r["due_date"] == P["due"]]
                if not ids:
                    return None
                time.sleep(PAUSE)
                return [("project_management.update_task", {"task_id": i, "field": "due_date", "new_value": P["new_due"]}) for i in ids]
            return None
        return policy
    search = ("project_management.search_tasks", {"task_name": P["src"], "assigned_to_email": P["assignee"]})

    def policy(messages):
        h = tool_history(messages)
        if not h or notified(messages):
            return [search]
        if h[-1][0] == search[0]:
            rows = [r for r in (parse_json(h[-1][2]) or []) if r["task_name"] == P["src"] and r["assigned_to_email"] == P["assignee"]]
            due = rows[0]["due_date"]
            created = [x for x in h if x[0] == "project_management.create_task"]
            if created:
                return [("project_management.update_task", {"task_id": clean(created[-1][2]), "field": "due_date", "new_value": due})]
            time.sleep(PAUSE)
            return [("project_management.create_task", {"task_name": P["follow"], "assigned_to_email": P["assignee"],
                                                        "list_name": P["lst"], "due_date": due, "board": P["board"]})]
        return None
    return policy


def calendar_lost_update(P, role):
    delta = P["d1"] if role == "A" else P["d2"]
    d = P["date"]
    search = ("calendar.search_events", {"query": P["ename"], "time_min": f"{d} 00:00:00", "time_max": f"{d} 23:59:59"})
    get = ("calendar.get_event_information_by_id", {"event_id": P["eid"], "field": "event_start"})

    def policy(messages):
        h = tool_history(messages)
        if not h:
            return [search]
        if notified(messages):
            rec = prefix_record(messages, "calendar_events", P["eid"])
            if rec:
                return [("calendar.update_event", {"event_id": P["eid"], "field": "event_start", "new_value": shift(rec["event_start"], delta)})]
            return [get]
        last_tool, _, last_res = h[-1]
        if last_tool == search[0]:
            rows = [r for r in (parse_json(last_res) or []) if r["event_id"] == P["eid"]]
            start = rows[0]["event_start"]
            time.sleep(PAUSE)
            return [("calendar.update_event", {"event_id": P["eid"], "field": "event_start", "new_value": shift(start, delta)})]
        if last_tool == get[0]:
            start = json.loads(clean(last_res))["event_start"]
            return [("calendar.update_event", {"event_id": P["eid"], "field": "event_start", "new_value": shift(start, delta)})]
        return None
    return policy


def crm_email_domain(P, role):
    dom, owner = P["domain"], P["owner"]
    if role == "A":
        search = ("customer_relationship_manager.search_customers", {"customer_email": dom})

        def policy(messages):
            h = tool_history(messages)
            if not h or notified(messages):
                return [search]
            if h[-1][0] == search[0]:
                todo = [(r["customer_id"], r["customer_email"].split("@")[0] + f"@{dom}.com")
                        for r in (parse_json(h[-1][2]) or []) if not r["customer_email"].endswith(f"@{dom}.com")]
                if not todo:
                    return None
                time.sleep(PAUSE)
                return [("customer_relationship_manager.update_customer", {"customer_id": i, "field": "customer_email", "new_value": e}) for i, e in todo]
            return None
        return policy
    search = ("customer_relationship_manager.search_customers", {"customer_name": P["ref_name"]})
    subj = f"New lead: {P['new_name']}"

    def policy(messages):
        h = tool_history(messages)
        if not h or notified(messages):
            return [search]
        last_tool, _, last_res = h[-1]
        if last_tool == search[0]:
            rows = [r for r in (parse_json(last_res) or []) if r["customer_name"] == P["ref_name"]]
            email = f"{P['new_local']}@{rows[0]['customer_email'].split('@')[1]}"
            added = [x for x in h if x[0] == "customer_relationship_manager.add_customer"]
            sent = any(x[0] == "email.send_email" for x in h)
            if added:
                calls = [("customer_relationship_manager.update_customer", {"customer_id": clean(added[-1][2]), "field": "customer_email", "new_value": email})]
                if not sent:
                    calls.append(("email.send_email", {"recipient": owner, "subject": subj, "body": f"Added {P['new_name']} ({email}) as a new lead."}))
                return calls
            time.sleep(PAUSE)
            return [("customer_relationship_manager.add_customer", {"customer_name": P["new_name"], "assigned_to_email": owner, "status": "Lead",
                                                                   "customer_email": email, "product_interest": "Consulting", "last_contact_date": "2023-11-30"})]
        if last_tool == "customer_relationship_manager.add_customer":
            return [("email.send_email", {"recipient": owner, "subject": subj, "body": f"Added {P['new_name']} ({h[-1][1]['customer_email']}) as a new lead."})]
        return None
    return policy


def double_booking(P, role):
    d, slot = P["date"], P["slot"]
    name, who = (P["n1"], P["p1"]) if role == "A" else (P["n2"], P["p2"])
    search = ("calendar.search_events", {"query": "", "time_min": f"{d} 00:00:00", "time_max": f"{d} 23:59:59"})

    def policy(messages):
        h = tool_history(messages)
        if not h or notified(messages):
            return [search]
        if h[-1][0] == search[0]:
            rows = parse_json(h[-1][2]) or []
            taken = any(r["event_start"] == f"{d} {slot}:00" for r in rows)
            mine = any(x[0] == "calendar.create_event" for x in h)
            if taken:
                if mine and not any(r["event_name"] == name for r in rows if r["event_start"] == f"{d} {slot}:00"):
                    return None
                if mine and sum(1 for r in rows if r["event_start"] == f"{d} {slot}:00") > 1:
                    return [("nexus.retract_my_actions", {"count": 1})] if role == "B" else None
                return None
            if mine:
                return None
            time.sleep(PAUSE)
            return [("calendar.create_event", {"event_name": name, "participant_email": who, "event_start": f"{d} {slot}:00", "duration": "30"})]
        return None
    return policy


def crm_notes_append(P, role):
    add = P["a1"] if role == "A" else P["a2"]
    search = ("customer_relationship_manager.search_customers", {"customer_name": P["cname"]})

    def policy(messages):
        h = tool_history(messages)
        if not h:
            return [search]
        if notified(messages):
            rec = prefix_record(messages, "crm_data", P["cid"])
            if rec:
                notes = (rec.get("notes") or "").rstrip()
                return [("customer_relationship_manager.update_customer", {"customer_id": P["cid"], "field": "notes", "new_value": (notes + " " + add).strip()})]
            return [search]
        if h[-1][0] == search[0]:
            rows = [r for r in (parse_json(h[-1][2]) or []) if r["customer_id"] == P["cid"]]
            notes = (rows[0].get("notes") or "").rstrip()
            if not any(x[0].endswith("update_customer") for x in h):
                time.sleep(PAUSE)
            return [("customer_relationship_manager.update_customer", {"customer_id": P["cid"], "field": "notes", "new_value": (notes + " " + add).strip()})]
        return None
    return policy


_T = {"crm_write_skew": crm_write_skew, "calendar_first_meeting": calendar_first_meeting, "pm_canary": pm_canary,
      "calendar_lost_update": calendar_lost_update, "crm_email_domain": crm_email_domain, "double_booking": double_booking,
      "crm_notes_append": crm_notes_append}
POLICIES = {name: (lambda role, P=P: _T[P["template"]](P, role)) for name, P in PARAMS.items()}
