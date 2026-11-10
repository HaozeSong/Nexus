"""Figures from the result files: python scripts/make_figures.py [--results results] [--out results/figures]

  fig_concurrency_<bench>.pdf  SCR, GPU time/task, tokens/task, mean JCT vs N for every protocol in results/conc_<bench>_*
  fig_length_<bench>.pdf       tokens and GPU time relative to serial vs chained length L (runs with params.long > 1)
  fig_replay.pdf               invalidation outcomes, decoded share and draft acceptance from results/replay_*
  fig_kv_<bench>.pdf           SCR and GPU time vs KV pool from results/kv_*
"""
import argparse
import collections
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"serial": "#777777", "twopl": "#eda100", "occ": "#1baf7a", "nexus_undo": "#e87ba4", "mtpo": "#eb6834", "nexus": "#1f5aa3",
       "nexus_nodraft": "#7fb0e6", "nexus_restart": "#999999", "nexus_notify": "#c2185b", "nexus_wait": "#5a0026", "nexus_proceed": "#2a78d6",
       "nexus_nointents": "#8e0038", "nexus_nopriority": "#6a3d9a", "nexus_poskeys": "#444444"}
LABEL = {"serial": "Serial", "twopl": "2PL", "occ": "OCC", "nexus_undo": "Undo", "mtpo": "MTPO", "nexus": "Nexus",
         "nexus_nodraft": "No-draft", "nexus_restart": "Restart", "nexus_notify": "Notify", "nexus_wait": "Always wait",
         "nexus_proceed": "Always proceed", "nexus_nointents": "No intents", "nexus_nopriority": "No priorities", "nexus_poskeys": "Position keys"}
ORDER = list(COL)

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/figures")
a = ap.parse_args()
R, OUT = Path(a.results), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)


def load(pattern):
    rows = []
    for d in sorted(R.glob(pattern)):
        f = d / "runs.jsonl"
        if f.exists():
            for line in f.open():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def metrics(r):
    nt = max(1, len(r["tasks"]))
    lat = [t["latency_s"] for t in r["tasks"].values() if "latency_s" in t]
    return {"success": 100 * r["pair_correct"] / max(1, r["pairs"]), "gpu": (r.get("gpu_s") or 0) / nt,
            "tokens": r["tokens_total"] / nt / 1000, "jct": st.mean(lat) if lat else 0}


def by_protocol_n(rows, long=1):
    by = collections.defaultdict(list)
    for r in rows:
        if int(r["params"].get("long", 1)) != long:
            continue
        by[(r["protocol_id"], r["n_agents"])].append(metrics(r))
    return by


def panel(ax, by, key, ylabel, title):
    protos = [p for p in ORDER if any(k[0] == p for k in by)]
    xs = sorted({k[1] for k in by})
    for p in protos:
        ys, es = [], []
        for n in xs:
            v = [m[key] for m in by.get((p, n), [])] or [m[key] for m in by.get((p, 2), [])] if p == "serial" else [m[key] for m in by.get((p, n), [])]
            ys.append(st.mean(v) if v else float("nan")); es.append((max(v) - min(v)) / 2 if len(v) > 1 else 0)
        if p == "serial":
            ax.axhline(ys[0], color=COL[p], ls="--", lw=1, label=LABEL[p])
        else:
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=1.5, color=COL[p], label=LABEL[p], capsize=2)
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("Concurrent agents N"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)


def fig_concurrency():
    for d in sorted(R.glob("conc_*")):
        bench = d.name.split("_")[1]
        rows = load(d.name)
        if not rows:
            continue
        by = by_protocol_n(rows)
        fig, axs = plt.subplots(1, 4, figsize=(13, 2.8))
        panel(axs[0], by, "success", "SCR (%)", f"{bench}: serializable completion rate")
        panel(axs[1], by, "gpu", "GPU time per Task (s)", f"{bench}: GPU cost")
        panel(axs[2], by, "tokens", "K Tokens per Task", f"{bench}: tokens")
        panel(axs[3], by, "jct", "JCT (s)", f"{bench}: job completion time")
        axs[0].legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(OUT / f"fig_concurrency_{bench}.pdf"); plt.close(fig)
        print("wrote", OUT / f"fig_concurrency_{bench}.pdf")


def fig_length():
    for d in sorted(R.glob("conc_*")):
        bench = d.name.split("_")[1]
        rows = [r for r in load(d.name) if int(r["params"].get("long", 1)) >= 1]
        Ls = sorted({int(r["params"].get("long", 1)) for r in rows})
        if len(Ls) < 2:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
        for key, ax, yl in (("tokens", axs[0], "Token cost (x serial)"), ("gpu", axs[1], "GPU time (x serial)")):
            for p in [p for p in ORDER if p != "serial" and any(r["protocol_id"] == p for r in rows)]:
                ys = []
                for L in Ls:
                    ser = [metrics(r)[key] for r in rows if r["protocol_id"] == "serial" and int(r["params"].get("long", 1)) == L]
                    v = [metrics(r)[key] for r in rows if r["protocol_id"] == p and int(r["params"].get("long", 1)) == L]
                    ys.append((st.mean(v) / st.mean(ser)) if v and ser and st.mean(ser) else float("nan"))
                ax.plot(Ls, ys, marker="o", ms=4, color=COL.get(p, "k"), label=LABEL.get(p, p))
            ax.axhline(1.0, color="#777", ls="--", lw=1); ax.set_xlabel("Chained tasks L"); ax.set_ylabel(yl); ax.set_xticks(Ls); ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / f"fig_length_{bench}.pdf"); plt.close(fig)
        print("wrote", OUT / f"fig_length_{bench}.pdf")


def fig_replay():
    f = R / "replay_study_summary.jsonl"
    if not f.exists():
        return
    rows = [json.loads(l) for l in f.open()]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["benchmark"], r["protocol"])].append(r)
    benches = sorted({k[0] for k in by})
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
    for i, b in enumerate(benches):
        v = by.get((b, "nexus"), [])
        ver = sum(x["draft_verified"] for x in v); acc = sum(x["draft_accepted"] for x in v); dec = sum(x["draft_decoded"] for x in v)
        axs[0].bar(i, 100 * acc / max(1, ver), color=COL["nexus"]); axs[1].bar(i, 100 * dec / max(1, acc + dec), color=COL["nexus"])
    for ax, yl in ((axs[0], "Draft tokens accepted (%)"), (axs[1], "Suffix decoded again (%)")):
        ax.set_xticks(range(len(benches))); ax.set_xticklabels(benches); ax.set_ylabel(yl); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_replay.pdf"); plt.close(fig)
    print("wrote", OUT / "fig_replay.pdf")


def fig_kv():
    groups = collections.defaultdict(dict)
    for d in sorted(R.glob("kv_*")):
        _, pool, bench, *_ = d.name.split("_")
        groups[bench][pool] = by_protocol_n(load(d.name))
    for bench, pools in groups.items():
        xs = sorted(pools, key=lambda s: int(s.rstrip("k")) if s.rstrip("k").isdigit() else 0, reverse=True)
        fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
        for key, ax, yl in (("success", axs[0], "SCR (%)"), ("gpu", axs[1], "GPU time per Task (s)")):
            protos = [p for p in ORDER if any(k[0] == p for pool in xs for k in pools[pool])]
            for p in protos:
                ys = [st.mean([m[key] for m in pools[pool].get((p, 8), [])] or [float("nan")]) for pool in xs]
                ax.plot(range(len(xs)), ys, marker="o", ms=4, color=COL.get(p, "k"), label=LABEL.get(p, p))
            ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs); ax.set_xlabel("KV-cache pool (tokens)"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / f"fig_kv_{bench}.pdf"); plt.close(fig)
        print("wrote", OUT / f"fig_kv_{bench}.pdf")


fig_concurrency(); fig_length(); fig_replay(); fig_kv()
