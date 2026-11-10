"""Protocol matrix on scripted (no-LLM) agents: one lost-update cell per short suite; naive must fail, the coordinating protocols
and Nexus (and its repair ablations) must keep the pair serializable."""
from __future__ import annotations

import importlib
import unittest

from benchmarks.registry import resolve
from nexus.engine.scripted import ScriptedLLM
from nexus.runner import run_trial

CELLS = ["tpcc/neworder_vs_derived_restock", "tpcc/restock_then_half_order", "micro/micro_r4_w2_p5_q5_1"]
OK = ["serial", "twopl", "occ", "mtpo", "nexus", "nexus_nodraft", "nexus_notify", "nexus_restart", "nexus_nointents", "nexus_proceed", "nexus_wait"]


def trial(cell, proto):
    pol = importlib.import_module(f"benchmarks.{cell.benchmark}.scripted").POLICIES[cell.name]
    return run_trial(cell, proto, {r: ScriptedLLM(pol(r)) for r in "AB"}, 0, max_steps=40)


class TestProtocols(unittest.TestCase):
    def test_coordinating_protocols_are_serializable(self):
        for c in resolve(CELLS):
            for p in OK:
                rec = trial(c, p)
                self.assertTrue(rec["correct"], f"{c.name} {p}: {rec['detail']}")

    def test_naive_loses_updates(self):
        self.assertTrue(any(not trial(c, "naive")["correct"] for c in resolve(CELLS)))

    def test_nexus_invalidates_instead_of_aborting(self):
        for c in resolve(CELLS[:1]):
            rec = trial(c, "nexus")
            self.assertEqual(rec["stats"]["aborts"], 0)
            self.assertGreaterEqual(rec["stats"].get("invalidations", 0) + rec["stats"].get("notifications", 0), 1)


if __name__ == "__main__":
    unittest.main()
