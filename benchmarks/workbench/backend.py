"""WorkBench (Styles et al. 2024) as a Backend: pandas tables + the benchmark's own tool functions."""
from __future__ import annotations

import os
import threading

from nexus.backend import Backend
from nexus.paths import vendored
from nexus.tools import ToolSpec, schemas_from

WORKBENCH = vendored("WorkBench")

from src.tools import analytics, calendar, company_directory  # noqa: E402
from src.tools import customer_relationship_manager as crm  # noqa: E402
from src.tools import email, project_management as pm  # noqa: E402
from src.tools import state as wb_state  # noqa: E402
from src.tools.tool import tool_to_openai_schema  # noqa: E402
from src.tools.toolkits import all_tools  # noqa: E402

TOOLS = {t.name: t for t in all_tools}
ID_COLS = {"calendar_events": "event_id", "emails": "email_id", "project_tasks": "task_id", "crm_data": "customer_id"}
TABLE_OF = {"calendar": "calendar_events", "email": "emails", "project_management": "project_tasks",
            "customer_relationship_manager": "crm_data", "analytics": "analytics_data", "company_directory": "directory_emails"}

# WorkBench tools fetch state via a thread-local get_state() and allocate ids via generate_next_id(); patch
# both so a tool runs against the state we hand it and ids are stable across replays.
_tls = threading.local()


def _get_state():
    return _tls.state


def _generate_next_id(df, id_col):
    table = next(t for t, c in ID_COLS.items() if c == id_col)
    return _tls.op.next_id(table)


for _m in (calendar, email, pm, crm, analytics, company_directory):
    _m.get_state = _get_state
    if hasattr(_m, "generate_next_id"):
        _m.generate_next_id = _generate_next_id

_cwd_lock = threading.Lock()


def _pristine():
    with _cwd_lock:
        old = os.getcwd()
        os.chdir(WORKBENCH)
        try:
            return wb_state._pristine_state()
        finally:
            os.chdir(old)


PRISTINE = _pristine()


def _spec(name, kind, id_param=None, **kw):
    table = TABLE_OF[name.split(".")[0]]
    return ToolSpec(name, kind, table, id_param, id_col=ID_COLS.get(table), **kw)


SPECS = {s.name: s for s in [
    _spec("calendar.get_event_information_by_id", "read", "event_id"),
    _spec("calendar.search_events", "read"),
    _spec("calendar.create_event", "write", range_write=True),
    _spec("calendar.update_event", "write", "event_id"),
    _spec("calendar.delete_event", "write", "event_id"),
    _spec("email.get_email_information_by_id", "read", "email_id"),
    _spec("email.search_emails", "read"),
    _spec("email.send_email", "write", range_write=True, unrecoverable=True),
    _spec("email.reply_email", "write", "email_id", range_write=True, unrecoverable=True),
    _spec("email.forward_email", "write", "email_id", range_write=True, unrecoverable=True),
    _spec("email.delete_email", "write", "email_id"),
    _spec("project_management.get_task_information_by_id", "read", "task_id"),
    _spec("project_management.search_tasks", "read"),
    _spec("project_management.create_task", "write", range_write=True),
    _spec("project_management.update_task", "write", "task_id"),
    _spec("project_management.delete_task", "write", "task_id"),
    _spec("customer_relationship_manager.search_customers", "read"),
    _spec("customer_relationship_manager.update_customer", "write", "customer_id"),
    _spec("customer_relationship_manager.add_customer", "write", range_write=True),
    _spec("customer_relationship_manager.delete_customer", "write", "customer_id"),
    _spec("analytics.get_visitor_information_by_id", "read"),
    _spec("analytics.engaged_users_count", "read"),
    _spec("analytics.traffic_source_count", "read"),
    _spec("analytics.total_visits_count", "read"),
    _spec("analytics.get_average_session_duration", "read"),
    _spec("company_directory.find_email_address", "read"),
]}
# analytics.create_plot writes image files; excluded.

_DESCRIPTIONS = {n: (tool_to_openai_schema(TOOLS[n])["function"]["description"].strip(),
                     tool_to_openai_schema(TOOLS[n])["function"]["parameters"]) for n in SPECS}


class WorkBenchBackend(Backend):
    name = "workbench"
    id_cols = ID_COLS
    specs = SPECS

    def fresh_base(self):
        return PRISTINE.copy()

    def copy(self, state):
        return state.copy()

    def run_tool(self, state, tool, kwargs, op):
        _tls.state, _tls.op = state, op
        try:
            return str(TOOLS[tool](**kwargs))
        finally:
            _tls.op = None

    def initial_next_ids(self, base):
        return {t: int(getattr(base, t)[c].max()) + 1 for t, c in ID_COLS.items()}

    def record(self, state, table, rid):
        df = getattr(state, table)
        rows = df[df[ID_COLS[table]] == rid].to_dict(orient="records")
        return rows[0] if rows else None

    def tool_schemas(self, style, names=None):
        return schemas_from(_DESCRIPTIONS, style, names)
