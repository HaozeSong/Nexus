"""Anchored replay end to end on the simulated token-level engine: two scripted agents on a conflicting TPC-C cell, the Nexus
manager with forward validation cancelling the in-flight generation, and the replay reusing the draft."""
from __future__ import annotations

import importlib
import unittest

from benchmarks.registry import resolve
from nexus.engine.sim import SimEngine
from nexus.runner import run_trial


def _cell(name):
    return resolve([name])[0]


class TestReplaySim(unittest.TestCase):
    def _run(self, proto, cell_name, gen_time=0.6, eps=0.0):
        cell = _cell(cell_name)
        pol = importlib.import_module(f"benchmarks.{cell.benchmark}.scripted").POLICIES[cell.name]
        llms = {r: SimEngine(pol(r), gen_time=gen_time, eps=eps) for r in "AB"}
        return run_trial(cell, proto, llms, 0, max_steps=40), llms

    def test_nexus_is_correct_and_reuses_the_draft(self):
        rec, llms = self._run("nexus", "tpcc/neworder_vs_derived_restock")
        self.assertTrue(rec["correct"], rec["detail"])
        calls = rec["calls"]
        self.assertGreaterEqual(rec["preemptions"] + rec["stats"].get("invalidations", 0), 1)
        # at least one turn after the invalidation was decoded against a draft and accepted part of it
        drafted = [c for c in calls if c.get("draft_verified", 0) > 0]
        self.assertTrue(drafted, "no turn used a draft")
        self.assertTrue(any(c["draft_accepted"] > 0 for c in drafted))

    def test_regeneration_without_drafts_matches_replay_output(self):
        r1, _ = self._run("nexus", "tpcc/restock_then_half_order")
        r2, _ = self._run("nexus_nodraft", "tpcc/restock_then_half_order")
        self.assertTrue(r1["correct"] and r2["correct"])
        self.assertEqual(r1["ops"][-1]["op"].split("(")[0], r2["ops"][-1]["op"].split("(")[0])
        self.assertTrue(all(c.get("draft_verified", 0) == 0 for c in r2["calls"]))

    def test_restart_and_notify_ablations_run(self):
        for p in ("nexus_restart", "nexus_notify"):
            rec, _ = self._run(p, "tpcc/neworder_vs_derived_restock")
            self.assertEqual(rec["agent_status"], {"A": "ok", "B": "ok"}, p)

    def test_naive_loses_the_update(self):
        rec, _ = self._run("naive", "tpcc/neworder_vs_derived_restock", gen_time=0.3)
        self.assertFalse(rec["correct"])


if __name__ == "__main__":
    unittest.main()
