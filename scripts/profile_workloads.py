"""Workload profiling (agent tasks as transactions): per-task rounds, read/write set sizes, positions of
reads/writes along the trajectory, and task length, across 8 benchmarks (3 real/public traces, 1 official benchmark trajectory set,
4 of ours). Output: results/profiles/profiles.json (per-task metrics), results/profiles/profile_summary.json, and the figure.
python scripts/profile_workloads.py [--fig <path.pdf>]"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

from nexus.paths import ROOT, THIRD_PARTY
RAW = ROOT / "results/profiles/raw"
READONLY = {"cat", "ls", "grep", "rg", "find", "head", "tail", "wc", "which", "pwd", "tree", "stat", "file", "less", "more", "type",
            "diff", "sort", "uniq", "cut", "column", "jq", "awk", "nl", "od", "xxd", "realpath", "basename", "dirname", "du", "df", "env", "printenv"}
WRITERS = {"sed", "tee", "cp", "mv", "rm", "mkdir", "touch", "chmod", "chown", "ln", "rmdir", "patch", "dd", "truncate", "apply_patch", "git_apply"}
PATH_RE = re.compile(r"(?<![\w@])((?:\.{0,2}/)?(?:[\w.-]+/)+[\w.-]+|[\w-]+\.(?:py|js|ts|json|md|txt|yaml|yml|toml|cfg|ini|sh|c|h|cpp|rs|go|java|html|css|csv|sql|lock))")


def classify_cmd(cmd: str):
    """Classify a shell command as read / write / exec by its executables and redirections."""
    if not cmd:
        return "exec"
    if re.search(r"(^|[^<>])>{1,2}\s*\S", cmd) or " tee " in f" {cmd} " or re.search(r"\bsed\s+-i", cmd) or "apply_patch" in cmd or re.search(r"cat\s*>\s*", cmd):
        return "write"
    parts = re.split(r"\s*(?:\|\||&&|;|\|)\s*", cmd.strip())
    execs = []
    for p in parts:
        toks = p.strip().split()
        while toks and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]) or toks[0] in ("sudo", "time", "nohup", "env")):
            toks = toks[1:]
        if toks:
            execs.append(toks[0].split("/")[-1])
    if not execs:
        return "exec"
    if any(e in WRITERS for e in execs):
        return "write"
    if all(e in READONLY or e == "cd" or e == "echo" for e in execs):
        return "read"
    return "exec"


def objs_in(text: str):
    return set(m.group(1) for m in PATH_RE.finditer(text or "") if "/" in m.group(1) or "." in m.group(1))


def finish(calls, dur=None, tokens=None):
    """calls: list of (kind, obj) in order, kind in read/write/exec/other. Returns per-task metrics."""
    acc = [(k, o) for k, o in calls if k in ("read", "write", "exec")]
    n = len(acc)
    if n == 0:
        return None
    reads = [i for i, (k, _) in enumerate(acc) if k == "read"]
    writes = [i for i, (k, _) in enumerate(acc) if k == "write"]
    rset = set(o for k, o in acc if k == "read" and o)
    wset = set(o for k, o in acc if k == "write" and o)
    pos = lambda i: i / max(1, n - 1) if n > 1 else 0.0
    fw = min(writes) if writes else None
    return {"n_calls": n, "n_read": len(reads), "n_write": len(writes), "n_exec": n - len(reads) - len(writes),
            "read_set": len(rset) if rset else None, "write_set": len(wset) if wset else None,
            "read_pos": [pos(i) for i in reads], "write_pos": [pos(i) for i in writes],
            "first_write_pos": pos(fw) if fw is not None else None,
            "reads_after_first_write": (sum(1 for i in reads if fw is not None and i > fw) / len(reads)) if reads and fw is not None else (0.0 if reads else None),
            "dur_s": dur, "tokens": tokens}


# ---------------- ours ----------------
OURS_READ = re.compile(r"\.(get_|search_|list_|read_|grep|run_tests|calculate|think|find_|calculate)")
def ours(bench):
    out = []
    for t in json.load(open(ROOT / f"results/profiles/serial_{bench}.json")):
        calls = []
        for c in t["calls"]:
            tool, a = c["tool"], c.get("args") or {}
            kind = "read" if OURS_READ.search(tool) else "write"
            if bench == "workspace":
                obj = a.get("path") or a.get("test_file") or ("*" if tool.endswith(("list_files", "grep")) else None)
                if tool.endswith("run_tests"):
                    kind = "exec"
            else:
                obj = None
                for k in ("reservation_id", "user_id", "customer_id", "event_id", "task_id", "order_id", "item_id", "email_id"):
                    if k in a:
                        obj = f"{tool.split('.')[0]}:{a[k]}"; break
                if obj is None:
                    obj = f"{tool.split('.')[0]}:*" if kind == "read" else f"{tool.split('.')[0]}:new"
            calls.append((kind, obj))
        u = t.get("usage") or {}
        tok = (u.get("input", 0) + u.get("output", 0) + u.get("cache_read", 0)) if u else None
        m = finish(calls, t.get("latency_s"), tok)
        if m:
            out.append(m)
    return out


# ---------------- WorkBench official trajectories ----------------
def workbench_official():
    import csv
    out = []
    for r in csv.DictReader(open(ROOT / "results/workbench_trajectory_stats.csv")):
        n = int(r["n"]); nw = int(r["n_write"]); nr = int(r["n_read"])
        if n == 0:
            continue
        fw = float(r["first_write_pos"]) if r["first_write_pos"] not in ("", "nan") and nw else None
        lw = float(r["last_write_pos"]) if r["last_write_pos"] not in ("", "nan") and nw else None
        rb = int(r["reads_before_first_write"]) if r["reads_before_first_write"] not in ("", "nan") else nr
        out.append({"n_calls": n, "n_read": nr, "n_write": nw, "n_exec": 0, "read_set": None, "write_set": None,
                    "read_pos": [], "write_pos": [p for p in (fw, lw) if p is not None], "first_write_pos": fw,
                    "reads_after_first_write": ((nr - rb) / nr) if nr and fw is not None else (0.0 if nr else None), "dur_s": None, "tokens": None})
    return out


# ---------------- TraceLab (real Claude Code / Codex sessions) ----------------
def tracelab():
    import duckdb
    con = duckdb.connect(str(THIRD_PARTY / "TraceLab/trace/syfi_coding_trace.duckdb"), read_only=True)
    rows = con.execute("""
        with r as (select session_id, round_index, round_pk, first_input_event_type, output_tokens, input_tokens_total,
                          sum(case when first_input_event_type='user_message' then 1 else 0 end) over (partition by session_id order by round_index) as req
                   from rounds)
        select r.session_id, r.req, r.round_index, t.tool_index, t.tool_name, t.executables, t.emitted_at, t.result_at, r.output_tokens
        from r left join tool_calls t on t.round_pk = r.round_pk
        order by r.session_id, r.round_index, t.tool_index""").fetchall()
    by = collections.defaultdict(list)
    for sid, req, ri, ti, tool, execs, t0, t1, otok in rows:
        by[(sid, req)].append((tool, execs, t0, t1, otok, ri))
    out = []
    for key, items in by.items():
        calls = []; ts = []; otok = {}
        for tool, execs, t0, t1, ot, ri in items:
            otok[ri] = ot or 0
            if tool is None:
                continue
            if t0 is not None: ts.append(t0)
            if t1 is not None: ts.append(t1)
            if tool in ("Read", "Grep", "Glob", "LS", "WebFetch", "WebSearch", "view_image", "ToolSearch"):
                calls.append(("read", None))
            elif tool in ("Edit", "Write", "MultiEdit", "NotebookEdit", "apply_patch"):
                calls.append(("write", None))
            elif tool in ("Bash", "exec_command", "shell", "shell_command", "exec"):
                ex = [e.split("/")[-1] for e in (execs or [])]
                # shell writes (sed -i, redirects) are not identifiable from the released trace (no command text): a lower bound on
                # writes, consistent with the TraceLab numbers cited in the text (explicit edit tools only)
                if ex and all(e in READONLY or e in ("cd", "echo") for e in ex):
                    calls.append(("read", None))
                else:
                    calls.append(("exec", None))
            else:
                continue  # planning / agent-management tools are not shared-state accesses
        dur = (max(ts) - min(ts)).total_seconds() if len(ts) >= 2 else None
        m = finish(calls, dur, sum(otok.values()) or None)
        if m:
            out.append(m)
    return out


# ---------------- SWE-agent trajectories (nebius/SWE-agent-trajectories, SWE-bench) ----------------
def swe_agent():
    out = []
    for f in sorted(RAW.glob("swe_*.json")):
        for r in json.load(open(f))["rows"]:
            traj = r["row"]["trajectory"]
            if isinstance(traj, str):
                try:
                    traj = json.loads(traj)
                except Exception:
                    import ast
                    traj = ast.literal_eval(traj)
            calls = []; cur = None
            for m in traj:
                if m.get("role") != "ai":
                    continue
                blocks = re.findall(r"```(?:\w*\n)?(.*?)```", m.get("text", ""), re.S)
                if not blocks:
                    continue
                cmd = blocks[-1].strip()
                head = cmd.split("\n", 1)[0].strip()
                toks = head.split()
                if not toks:
                    continue
                c = toks[0]
                if c in ("open", "cat"):
                    cur = toks[1] if len(toks) > 1 else cur; calls.append(("read", cur))
                elif c in ("goto", "scroll_down", "scroll_up"):
                    calls.append(("read", cur))
                elif c in ("search_dir", "search_file", "find_file", "ls", "grep", "find", "head", "tail", "wc", "tree"):
                    calls.append(("read", toks[-1] if c in ("search_file", "find_file") and len(toks) > 2 else "*"))
                elif c in ("edit", "insert", "str_replace", "str_replace_editor"):
                    calls.append(("write", cur))
                elif c == "create":
                    cur = toks[1] if len(toks) > 1 else cur; calls.append(("write", cur))
                elif re.search(r"\bsed\s+-i|>{1,2}\s*\S|\btee\b", head):
                    calls.append(("write", (list(objs_in(head)) or [cur])[0]))
                elif c in ("submit", "exit"):
                    continue
                else:
                    calls.append(("exec", None))
            m = finish(calls, None, None)
            if m:
                out.append(m)
    return out


# ---------------- Terminal-Bench trajectories (yoonholee/terminalbench-trajectories) ----------------
def terminal_bench():
    out = []
    for f in sorted(RAW.glob("tb_*.json")):
        for r in json.load(open(f))["rows"]:
            row = r["row"]
            if row.get("steps") in (None, "null", ""):
                continue
            steps = row["steps"] if isinstance(row["steps"], list) else json.loads(row["steps"])
            calls = []
            for s in steps:
                for t in (s.get("tools") or []):
                    fn, cmd = (t.get("fn") or ""), (t.get("cmd") or "")
                    if isinstance(cmd, (list, tuple)):
                        cmd = " ; ".join(str(c) for c in cmd)
                    elif not isinstance(cmd, str):
                        cmd = json.dumps(cmd)
                    if fn in ("bash_command", "execute_bash", "shell", "Bash", "run_shell_command", "execute"):
                        k = classify_cmd(cmd); calls.append((k, (list(objs_in(cmd)) or [None])[0] if k != "exec" else None))
                    elif fn in ("Read", "read_file", "view", "Glob", "Grep", "LS"):
                        calls.append(("read", (list(objs_in(cmd)) or [None])[0]))
                    elif fn in ("Edit", "Write", "write_file", "replace", "str_replace_editor", "apply_patch", "MultiEdit"):
                        calls.append(("write", (list(objs_in(cmd)) or [None])[0]))
                    else:
                        continue
            dur = float(row["duration_seconds"]) if row.get("duration_seconds") not in (None, "None", "") else None
            tok = None
            if row.get("input_tokens") not in (None, "None", ""):
                tok = float(row["input_tokens"]) + float(row.get("output_tokens") or 0)
            m = finish(calls, dur, tok)
            if m:
                out.append(m)
    return out


BENCHES = [("TraceLab\n(real coding\nsessions)", tracelab), ("SWE-agent\n(SWE-bench)", swe_agent), ("Terminal-\nBench", terminal_bench),
           ("WorkBench\n(official\ntrajectories)", workbench_official), ("Workspace\n(ours)", lambda: ours("workspace")),
           ("τ²-long\n(ours)", lambda: ours("tau2long")), ("TPC-C-long\n(ours)", lambda: ours("tpcclong")), ("τ²-airline\ncells (ours)", lambda: ours("tau2"))]


def q(xs, p):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, p)) if xs else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fig", default=str(ROOT / "results/figures/fig_profile.pdf")); a = ap.parse_args()
    data = {}
    for name, fn in BENCHES:
        try:
            data[name] = fn(); print(f"{name.replace(chr(10), ' '):40s} {len(data[name])} tasks", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAILED", name.replace(chr(10), ' '), e); data[name] = []
    (ROOT / "results/profiles").mkdir(exist_ok=True)
    json.dump(data, open(ROOT / "results/profiles/profiles.json", "w"))
    summ = {}
    for name, ts in data.items():
        if not ts: continue
        rp = [p for t in ts for p in t["read_pos"]]; wp = [p for t in ts for p in t["write_pos"]]
        summ[name.replace("\n", " ")] = {
            "tasks": len(ts), "calls_p50": q([t["n_calls"] for t in ts], 50), "calls_p90": q([t["n_calls"] for t in ts], 90),
            "read_share": sum(t["n_read"] for t in ts) / max(1, sum(t["n_calls"] for t in ts)),
            "write_share": sum(t["n_write"] for t in ts) / max(1, sum(t["n_calls"] for t in ts)),
            "tasks_with_write": sum(1 for t in ts if t["n_write"]) / len(ts),
            "read_set_p50": q([t["read_set"] for t in ts], 50), "write_set_p50": q([t["write_set"] for t in ts], 50),
            "first_write_pos_p50": q([t["first_write_pos"] for t in ts], 50), "first_write_pos_p25": q([t["first_write_pos"] for t in ts], 25), "first_write_pos_p75": q([t["first_write_pos"] for t in ts], 75),
            "read_pos_p50": q(rp, 50), "write_pos_p50": q(wp, 50),
            "reads_after_first_write": (lambda v: sum(v) / len(v) if v else None)([t["reads_after_first_write"] for t in ts if t["reads_after_first_write"] is not None and t["n_write"]]),
            "dur_p50": q([t["dur_s"] for t in ts], 50), "dur_p90": q([t["dur_s"] for t in ts], 90), "tokens_p50": q([t["tokens"] for t in ts], 50)}
    json.dump(summ, open(ROOT / "results/profiles/profile_summary.json", "w"), indent=1)
    for k, v in summ.items():
        print(k, {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    plot(data, a.fig)


def plot(data, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n, _ in BENCHES if data.get(n)]
    ext = 4  # first four are external
    c_ext, c_ours = "#4C72B0", "#DD8452"
    cols = [c_ext if i < ext else c_ours for i in range(len(names))]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 2.9))
    plt.rcParams.update({"font.size": 8})
    x = np.arange(len(names))
    SHORT = {"TraceLab": "TraceLab", "SWE-agent": "SWE-agent", "Terminal-": "Terminal-Bench", "WorkBench": "WorkBench", "Workspace": "Workspace*", "τ²-long": "τ²-long*", "TPC-C-long": "TPC-C-long*", "τ²-airline": "τ² cells*"}
    short = [SHORT.get(n.split("\n")[0], n.split("\n")[0]) for n in names]
    def box(ax, series, positions, color, width=0.6):
        b = ax.boxplot(series, positions=positions, widths=width, showfliers=False, patch_artist=True, medianprops=dict(color="black", lw=1.2), whis=(10, 90))
        for p in b["boxes"]:
            p.set_facecolor(color); p.set_alpha(0.75); p.set_edgecolor("black"); p.set_linewidth(0.6)
        return b
    # (a) rounds per task
    ax = axes[0]
    for i, n in enumerate(names):
        box(ax, [[t["n_calls"] for t in data[n]]], [i], cols[i])
    ax.set_yscale("log"); ax.set_ylabel("tool calls per task"); ax.set_title("(a) tool calls per task", fontsize=9, loc="left")
    # (b) read/write set sizes (distinct objects; call counts where arguments are unavailable)
    ax = axes[1]
    for i, n in enumerate(names):
        ts = data[n]
        rs = [t["read_set"] for t in ts if t["read_set"]] or [t["n_read"] for t in ts if t["n_read"]]
        ws = [t["write_set"] for t in ts if t["write_set"]] or [t["n_write"] for t in ts if t["n_write"]]
        hatch = "//" if not any(t["read_set"] for t in ts) else None
        b1 = box(ax, [rs], [i - 0.18], "#8fbfe0", 0.32); b2 = box(ax, [ws], [i + 0.18], "#e39c8f", 0.32)
        if hatch:
            for p in b1["boxes"] + b2["boxes"]: p.set_hatch(hatch)
    ax.set_yscale("log"); ax.set_ylabel("objects per task"); ax.set_title("(b) read-set (blue) / write-set (red) size", fontsize=9, loc="left")
    ax.text(0.02, 0.97, "hatched: call counts (no object ids in trace)", transform=ax.transAxes, fontsize=6, va="top")
    # (c) where reads and writes happen
    ax = axes[2]
    for i, n in enumerate(names):
        ts = data[n]
        rp = [p for t in ts for p in t["read_pos"]]; wp = [p for t in ts for p in t["write_pos"]]
        if rp: box(ax, [rp], [i - 0.18], "#8fbfe0", 0.32)
        if wp: box(ax, [wp], [i + 0.18], "#e39c8f", 0.32)
        fw = [t["first_write_pos"] for t in ts if t["first_write_pos"] is not None]
        if fw: ax.plot([i + 0.18], [np.median(fw)], marker="v", color="black", ms=4)
        raw = [t["reads_after_first_write"] for t in ts if t["reads_after_first_write"] is not None and t["n_write"]]
        if raw: ax.text(i, 1.04, f"{100*np.mean(raw):.0f}%", ha="center", fontsize=6.5)
    ax.set_ylim(-0.02, 1.2); ax.set_ylabel("position in task (0 = start, 1 = end)"); ax.set_title("(c) position of reads / writes (▼ median first write)", fontsize=9, loc="left")
    ax.text(0.5, 1.115, "% = share of reads issued after the first write", transform=ax.transAxes, fontsize=6.5, ha="center")
    # (d) task length in time (and tokens where time is unavailable)
    ax = axes[3]
    for i, n in enumerate(names):
        ts = data[n]
        d = [t["dur_s"] for t in ts if t["dur_s"]]
        if d: box(ax, [d], [i], cols[i])
        else: ax.text(i, 1.5, "n/a", ha="center", fontsize=7, color="gray")
    ax.set_yscale("log"); ax.set_ylabel("task duration (s)"); ax.set_title("(d) task duration", fontsize=9, loc="left")
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=c_ext, alpha=0.75, label="public traces / benchmarks"), mpatches.Patch(color=c_ours, alpha=0.75, label="our workloads (*)")], fontsize=6.5, loc="lower right", frameon=False)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha="right", fontsize=7); ax.grid(axis="y", alpha=0.3); ax.set_xlim(-0.6, len(names) - 0.4)
    fig.tight_layout(w_pad=1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); fig.savefig(str(path).rsplit(".", 1)[0] + ".png", dpi=160, bbox_inches="tight")
    print("figure ->", path)


if __name__ == "__main__":
    main()
