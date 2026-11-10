"""Contended WorkBench cells, generated from anomaly templates instantiated on several records each.
The first instance of a template carries the template's name (crm_write_skew, ...); further instances add _2, _3, ...
Each invariant accepts either serial order."""
from __future__ import annotations

import json
import re

import pandas as pd

from nexus.cell import Cell, Check
from nexus.store import ObjectStore
from benchmarks.workbench.backend import ID_COLS, PRISTINE, WorkBenchBackend

TODAY = "2023-11-30"


def _rows(store, table):
    return getattr(store.materialize(), table)


def _row(df, table, rid):
    r = df[df[ID_COLS[table]] == rid].to_dict(orient="records")
    return r[0] if r else None


def _unchanged_except(store: ObjectStore, table: str, allowed: set[str], allow_new: bool = False) -> list[str]:
    col = ID_COLS[table]
    base = getattr(store.base, table).fillna("").astype(str)
    final = _rows(store, table).fillna("").astype(str)
    problems = []
    b = base.set_index(col)
    f = final.set_index(col)
    for rid in b.index:
        if rid in allowed:
            continue
        if rid not in f.index:
            problems.append(f"{table}:{rid} deleted")
        elif not b.loc[rid].equals(f.loc[rid]):
            problems.append(f"{table}:{rid} modified")
    new = [rid for rid in f.index if rid not in b.index and rid not in allowed]
    if new and not allow_new:
        problems.append(f"{table}: unexpected new rows {new}")
    return problems


CAL = PRISTINE.calendar_events
CRM = PRISTINE.crm_data
PM = PRISTINE.project_tasks
CELLS: dict[str, Cell] = {}
PARAMS: dict[str, dict] = {}


def _add(name, template, cell, params):
    cell.backend = WorkBenchBackend
    cell.benchmark = "workbench"
    cell.template = template
    CELLS[name] = cell
    PARAMS[name] = dict(params, template=template)
    # cells are keyed by the 8-digit record ids in their parameters
    cell.objects = set(re.findall(r"\b\d{8}\b", json.dumps(PARAMS[name], default=str))) or {f"wb:{name}"}


# ------------------------------------------------------------------ T1 write skew on a CRM predicate
def t1(name, assignee, status, interest, ids, new_assignee):
    who = assignee.split("@")[0].replace(".", " ").title()

    def inv(store) -> Check:
        df = _rows(store, "crm_data")
        rows = {rid: _row(df, "crm_data", rid) for rid in ids}
        if any(r is None for r in rows.values()):
            return False, "customer deleted"
        won = all(r["status"] == "Won" and r["assigned_to_email"] == assignee for r in rows.values())
        moved = all(r["status"] == status and r["assigned_to_email"] == new_assignee for r in rows.values())
        p = _unchanged_except(store, "crm_data", set(ids))
        if p:
            return False, "; ".join(p)
        if won:
            return True, "A·B (Won, same owner)"
        if moved:
            return True, f"B·A ({status}, reassigned)"
        return False, "write skew: " + str({k: (v["status"], v["assigned_to_email"]) for k, v in rows.items()})
    _add(name, "crm_write_skew", Cell(name, "write skew (two RMW on overlapping predicate)",
        task_a=f"In the CRM, all of {assignee}'s customers that are currently in the {status} stage and whose product "
               f"interest is {interest} have just closed. Update the status of every such customer to Won.",
        task_b=f"In the CRM, reassign all of {assignee}'s customers that are currently in the {status} stage and whose "
               f"product interest is {interest} to {new_assignee} (set assigned_to_email).",
        invariant=inv), dict(assignee=assignee, status=status, interest=interest, ids=ids, new_assignee=new_assignee, who=who))


_g = CRM[CRM.status.isin(["Proposal", "Qualified", "Lead"])].groupby(["assigned_to_email", "status", "product_interest"]).customer_id.apply(list)
_t1 = [(k, v) for k, v in _g.items() if 2 <= len(v) <= 3]
_owners = sorted(CRM.assigned_to_email.unique())
_t1_first = [(k, v) for k, v in _g.items() if k == ("lena.schmidt@atlas.com", "Proposal", "Training")]
_picked = _t1_first + [x for x in _t1 if x != _t1_first[0]][:7]
for i, ((assignee, status, interest), ids) in enumerate(_picked):
    other = next(o for o in _owners if o != assignee)
    t1("crm_write_skew" + ("" if i == 0 else f"_{i+1}"), assignee, status, interest, sorted(ids), other)


# ------------------------------------------------------------------ T2 stale read then delete (first meeting of a day)
def t2(name, date, e1, e2, dur):
    day = pd.Timestamp(date).strftime("%B %-d")

    def inv(store) -> Check:
        df = _rows(store, "calendar_events")
        a, b = _row(df, "calendar_events", e1), _row(df, "calendar_events", e2)
        p = _unchanged_except(store, "calendar_events", {e1, e2})
        if p:
            return False, "; ".join(p)
        if a is None and b and b["event_start"] == f"{date} 16:00:00":
            return True, f"A·B ({e1} cancelled, {e2} moved)"
        if b is None and a and a["event_start"] == f"{date} 16:00:00":
            return True, f"B·A ({e1} moved, {e2} cancelled)"
        return False, f"{e1}={a and a['event_start']} {e2}={b and b['event_start']}"
    _add(name, "calendar_first_meeting", Cell(name, "stale read then delete (lost effect)",
        task_a=f"Cancel my first meeting on {day}.",
        task_b=f"Move my first meeting on {day} to 16:00 on the same day, keeping its duration.",
        invariant=inv), dict(date=date, e1=e1, e2=e2))


_days = CAL.assign(day=CAL.event_start.str[:10]).groupby("day")
_t2 = []
for day, g in _days:
    g = g.sort_values("event_start")
    if len(g) >= 2 and day > TODAY and not g.event_start.str.endswith("16:00:00").any() and g.iloc[0].event_start[11:] < "12:00:00":
        _t2.append((day, g.iloc[0].event_id, g.iloc[1].event_id, g.iloc[0].duration))
_t2 = [x for x in _t2 if x[0] == "2023-12-13"] + [x for x in _t2 if x[0] not in ("2023-12-13",)][:7]
for i, (day, e1, e2, dur) in enumerate(_t2):
    t2("calendar_first_meeting" + ("" if i == 0 else f"_{i+1}"), day, e1, e2, dur)


# ------------------------------------------------------------------ T3 canary (range repair vs derived create)
def t3(name, assignee, lst, board, due, new_due, ids, src_name, follow):
    def inv(store) -> Check:
        df = _rows(store, "project_tasks")
        for rid in ids:
            t = _row(df, "project_tasks", rid)
            if not t:
                return False, "task deleted"
            if t["due_date"] != new_due:
                return False, f"A's repair incomplete: {rid} due {t['due_date']}"
        new = df[df["task_name"].str.strip().str.lower() == follow.lower()].to_dict(orient="records")
        if len(new) != 1:
            return False, f"expected exactly one follow-up task, got {len(new)}"
        n = new[0]
        p = _unchanged_except(store, "project_tasks", set(ids), allow_new=True)
        if p:
            return False, "; ".join(p)
        ok = n["assigned_to_email"] == assignee and n["list_name"] == lst and n["board"] == board and n["due_date"] == new_due
        return (True, "canary mirrors repaired date") if ok else (False, f"follow-up task wrong: {n}")
    _add(name, "pm_canary", Cell(name, "stale premise + range scan misses new object (canary)",
        task_a=f"All of {assignee}'s project tasks in the {lst} list that are due on {due} have slipped. Change the due "
               f"date of every such task to {new_due}.",
        task_b=f"Create a follow-up project task for {assignee} named '{follow}' on the '{board}' board in the {lst} list, "
               f"with the same due date as {assignee}'s existing task '{src_name}' (the one assigned to {assignee}).",
        invariant=inv), dict(assignee=assignee, lst=lst, board=board, due=due, new_due=new_due, ids=ids, src=src_name, follow=follow))


_g3 = PM[PM.list_name == "Backlog"].groupby(["assigned_to_email", "due_date", "board"]).task_id.apply(list)
_namecount = PM.task_name.value_counts()
_t3 = [(k, v) for k, v in _g3.items() if 2 <= len(v) <= 3 and k[1] != "2023-12-15"]
_t3 = [x for x in _t3 if x[0] == ("leila.azizi@atlas.com", "2023-11-27", "Front end")] + [x for x in _t3 if x[0][0] != "leila.azizi@atlas.com"][:6]
if not any(x[0][1] == "2023-11-27" for x in _t3):   # the first instance is due before TODAY and would be filtered out: add it explicitly
    _t3 = [(("leila.azizi@atlas.com", "2023-11-27", "Front end"), ["00000149", "00000151"])] + _t3
for i, ((assignee, due, board), ids) in enumerate(_t3[:7]):
    src = PM[PM.task_id == ids[0]].iloc[0].task_name
    t3("pm_canary" + ("" if i == 0 else f"_{i+1}"), assignee, "Backlog", board, due, "2023-12-15" if due != "2023-12-15" else "2023-12-20",
       sorted(ids), src, f"Review {src.lower()}" if i == 0 and src == "Add animation to carousel" and False else ("Review carousel animation" if i == 0 else f"Follow up on {src.lower()}"))


# ------------------------------------------------------------------ T4 lost update (two RMW on one field)
def t4(name, eid, ename, who, date, start, dur, d1, d2):
    day = pd.Timestamp(date).strftime("%B %-d")
    target = (pd.Timestamp(start) + pd.Timedelta(minutes=d1 + d2)).strftime("%Y-%m-%d %H:%M:%S")

    def inv(store) -> Check:
        df = _rows(store, "calendar_events")
        e = _row(df, "calendar_events", eid)
        p = _unchanged_except(store, "calendar_events", {eid})
        if p:
            return False, "; ".join(p)
        if e and e["event_start"] == target and e["duration"] == dur:
            return True, f"both delays applied ({target[11:16]})"
        return False, f"lost update: start={e and e['event_start']} duration={e and e['duration']}"
    w1 = "one hour" if d1 == 60 else f"{d1} minutes"
    _add(name, "calendar_lost_update", Cell(name, "lost update (two RMW on one field)",
        task_a=f"Push back my '{ename}' meeting with {who} on {day} by {w1}.",
        task_b=f"Push back the '{ename}' meeting with {who} on {day} by {d2} minutes.",
        invariant=inv), dict(eid=eid, ename=ename, who=who, date=date, start=start, d1=d1, d2=d2, target=target))


_cnt = CAL.groupby(["event_name", "participant_email", CAL.event_start.str[:10]]).size()
_uniq = {k for k, v in _cnt.items() if v == 1}
_t4 = [r for _, r in CAL.iterrows() if (r.event_name, r.participant_email, r.event_start[:10]) in _uniq and r.event_start[11:] <= "13:00:00"]
_t4 = [r for r in _t4 if r.event_id == "00000275"] + [r for r in _t4 if r.event_id != "00000275" and r.event_start[:10] > TODAY][:7]
for i, r in enumerate(_t4):
    t4("calendar_lost_update" + ("" if i == 0 else f"_{i+1}"), r.event_id, r.event_name, r.participant_email, r.event_start[:10], r.event_start, r.duration, 60, 30)


# ------------------------------------------------------------------ T5 canary + unrecoverable effect (email domain)
def t5(name, domain, ref_id, ref_name, owner, new_name, new_local):
    def inv(store) -> Check:
        crm = _rows(store, "crm_data")
        q = _row(crm, "crm_data", ref_id)
        if not q or not q["customer_email"].endswith(f"@{domain}.com"):
            return False, f"reference customer not fixed: {q and q['customer_email']}"
        fixed_ids = set(CRM[CRM.customer_email.fillna("").str.contains(domain)].customer_id)
        for rid in fixed_ids:
            r = _row(crm, "crm_data", rid)
            if not r or not r["customer_email"].endswith(f"@{domain}.com"):
                return False, f"{rid} not fixed"
        new = crm[crm["customer_name"].str.strip().str.lower() == new_name.lower()].to_dict(orient="records")
        if len(new) != 1:
            return False, f"expected one {new_name}, got {len(new)}"
        if new[0]["customer_email"] != f"{new_local}@{domain}.com":
            return False, f"{new_name} built on stale domain: {new[0]['customer_email']}"
        p = _unchanged_except(store, "crm_data", fixed_ids, allow_new=True)
        em = _rows(store, "emails")
        sent = em[(em["inbox/outbox"] == "outbox") & (em["sender/recipient"] == owner) & em["subject"].str.contains(new_name.split()[0], case=False, regex=False)]
        if len(sent) == 0:
            return False, f"no notification email to {owner}"
        p += _unchanged_except(store, "emails", set(), allow_new=True)
        if p:
            return False, "; ".join(p)
        return True, "new lead on repaired domain, email sent"
    _add(name, "crm_email_domain", Cell(name, "canary + unrecoverable effect (send_email)",
        task_a=f"Customer email addresses at {domain} are missing the '.com' suffix. Find every CRM customer whose email "
               f"contains '{domain}' and update their customer_email so that it ends with '@{domain}.com' (keep the local "
               f"part before the @ unchanged).",
        task_b=f"Add a new CRM customer named '{new_name}' from the same company as the existing customer {ref_name}. Use "
               f"exactly the same email domain as in {ref_name}'s CRM record (the customer_email should be "
               f"{new_local}@<{ref_name.split()[0]}'s domain>), assigned to {owner}, status Lead, product interest Consulting, "
               f"last contact date {TODAY}. Then send an email to {owner} with the subject 'New lead: {new_name}' telling "
               f"them the lead was added, including the new customer's email address.",
        invariant=inv), dict(domain=domain, ref_id=ref_id, ref_name=ref_name, owner=owner, new_name=new_name, new_local=new_local))


_dom = CRM.customer_email.fillna("").str.split("@").str[-1]
_domc = _dom[_dom != ""].value_counts()
_t5 = ["nanoforcerobotics"] + [d for d in _domc.index if d != "nanoforcerobotics" and _domc[d] == 1 and "." not in d][:6]
_names = [("Riley Chen", "riley.chen"), ("Morgan Lee", "morgan.lee"), ("Casey Kim", "casey.kim"), ("Jamie Park", "jamie.park"), ("Avery Singh", "avery.singh"), ("Drew Novak", "drew.novak"), ("Sam Ortiz", "sam.ortiz")]
for i, d in enumerate(_t5):
    ref = CRM[_dom == d].iloc[0]
    t5("crm_email_domain" + ("" if i == 0 else f"_{i+1}"), d, ref.customer_id, ref.customer_name, ref.assigned_to_email, *_names[i])


# ------------------------------------------------------------------ T6 double booking (write skew via existence check)
def t6(name, date, slot, n1, p1, n2, p2):
    day = pd.Timestamp(date).strftime("%B %-d")
    start = f"{date} {slot}:00"

    def inv(store) -> Check:
        df = _rows(store, "calendar_events")
        at = df[df.event_start == start]
        p = _unchanged_except(store, "calendar_events", set(), allow_new=True)
        if p:
            return False, "; ".join(p)
        if len(at) == 1:
            return True, f"one meeting at {slot} ({at.iloc[0].event_name})"
        return False, f"{len(at)} meetings at {slot} on {date} (double booking)" if len(at) > 1 else "no meeting scheduled"
    _add(name, "double_booking", Cell(name, "write skew via existence check (double booking)",
        task_a=f"If I have no meeting starting at {slot} on {day}, schedule a 30-minute meeting called '{n1}' with {p1} at {slot} on {day}. If that slot is already taken, do nothing.",
        task_b=f"If I have no meeting starting at {slot} on {day}, schedule a 30-minute meeting called '{n2}' with {p2} at {slot} on {day}. If that slot is already taken, do nothing.",
        invariant=inv), dict(date=date, slot=slot, n1=n1, p1=p1, n2=n2, p2=p2))


_people = sorted(CAL.participant_email.unique())
_t6 = []
for day, g in _days:
    if day > TODAY and len(g) >= 1 and not g.event_start.str.endswith("15:00:00").any() and not ((g.event_start.str[11:] <= "15:00:00") & ((pd.to_datetime(g.event_start) + pd.to_timedelta(g.duration.astype(int), unit="m")).dt.strftime("%H:%M:%S") > "15:00:00")).any():
        _t6.append(day)
for i, day in enumerate(_t6[:6]):
    t6("double_booking" + ("" if i == 0 else f"_{i+1}"), day, "15:00", "budget sync", _people[3], "design review", _people[7])


# ------------------------------------------------------------------ T7 notes append (lost update on a text field)
def t7(name, cid, cname, base_notes):
    a1, a2 = f"{TODAY}: Had a call.", f"{TODAY}: Sent proposal."

    def inv(store) -> Check:
        df = _rows(store, "crm_data")
        r = _row(df, "crm_data", cid)
        p = _unchanged_except(store, "crm_data", {cid})
        if p:
            return False, "; ".join(p)
        notes = (r or {}).get("notes") or ""
        if a1 in notes and a2 in notes and (base_notes.strip() == "" or base_notes.strip() in notes):
            return True, "both appended, original kept"
        return False, f"notes={notes[:80]!r} (lost update)"
    _add(name, "crm_notes_append", Cell(name, "lost update on a text field (read-append-write)",
        task_a=f"Append the note '{a1}' to the end of the CRM notes of customer {cname}. Keep the existing notes; the notes field should contain the old notes followed by the new note.",
        task_b=f"Append the note '{a2}' to the end of the CRM notes of customer {cname}. Keep the existing notes; the notes field should contain the old notes followed by the new note.",
        invariant=inv), dict(cid=cid, cname=cname, base_notes=base_notes, a1=a1, a2=a2))


_namec = CRM.customer_name.value_counts()
_t7 = [r for _, r in CRM.iterrows() if _namec[r.customer_name] == 1 and isinstance(r.notes, str) and 10 < len(r.notes) < 80][:6]
for i, r in enumerate(_t7):
    t7("crm_notes_append" + ("" if i == 0 else f"_{i+1}"), r.customer_id, r.customer_name, r.notes)

for _c in CELLS.values():
    _c.backend = WorkBenchBackend
    _c.benchmark = "workbench"
