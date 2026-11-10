"""Content-keyed sampling (nexus.sampling): PRF, keys, re-alignment and the verify/accept rule."""
from __future__ import annotations

import math
import random
import unittest

from nexus.sampling import (ContentKeyedSampler, KeyState, align, generate_chunk, gumbel_noise, murmur3_32, verify_chunk,
                            window_key)


class TestPRF(unittest.TestCase):
    def test_murmur_is_deterministic_and_32bit(self):
        a, b = murmur3_32(42, 7, 1234), murmur3_32(42, 7, 1234)
        self.assertEqual(a, b)
        self.assertTrue(0 <= a <= 0xFFFFFFFF)
        self.assertNotEqual(murmur3_32(42, 7, 1234), murmur3_32(42, 8, 1234))
        self.assertNotEqual(murmur3_32(42, 7, 1234), murmur3_32(43, 7, 1234))

    def test_known_vectors(self):
        # fixed by the algorithm (MurmurHash3_x86_32 over the blocks seed_lo, seed_hi, key, token; len 16); the engine plugin and
        # the client must agree on these
        self.assertEqual(murmur3_32(0, 0, 0), 0x8134CDF8)
        self.assertEqual(murmur3_32(42, 7, 1234), 0xBD758A2D)

    def test_gumbel_noise_is_finite(self):
        for t in range(1000):
            g = gumbel_noise(1, 2, t)
            self.assertTrue(math.isfinite(g))

    def test_gumbel_noise_is_standard_gumbel(self):
        xs = [gumbel_noise(9, k, 5) for k in range(20000)]
        mean = sum(xs) / len(xs)
        self.assertAlmostEqual(mean, 0.5772, delta=0.03)     # Euler-Mascheroni constant


class TestKeys(unittest.TestCase):
    def test_same_window_same_count_same_key(self):
        s = ContentKeyedSampler(seed=1, m=3)
        st = KeyState()
        self.assertEqual(s.key([1, 2, 3, 4, 5], st), s.key([9, 9, 3, 4, 5], st))        # only the last m tokens matter
        self.assertNotEqual(s.key([1, 2, 3, 4, 5], st), s.key([1, 2, 3, 4, 6], st))

    def test_occurrence_count_separates_repeats(self):
        s = ContentKeyedSampler(seed=1, m=2)
        st = KeyState()
        k1 = s.key([7, 8], st); s.note([7, 8], st)
        k2 = s.key([7, 8], st)
        self.assertNotEqual(k1, k2)       # a second occurrence of the same window gets fresh noise

    def test_position_keys_shift_with_length(self):
        s = ContentKeyedSampler(seed=1, m=2, position_keys=True)
        st = KeyState()
        self.assertNotEqual(s.key([1, 2, 3], st), s.key([0, 1, 2, 3], st))
        c = ContentKeyedSampler(seed=1, m=2)
        self.assertEqual(c.key([1, 2, 3], st), c.key([0, 1, 2, 3], st))


class TestSampling(unittest.TestCase):
    def test_sample_is_argmax_of_logp_plus_noise(self):
        s = ContentKeyedSampler(seed=3, m=4)
        cands = {10: math.log(0.5), 11: math.log(0.3), 12: math.log(0.2)}
        key = 12345
        best = max(cands, key=lambda v: cands[v] + gumbel_noise(3, key, v))
        self.assertEqual(s.sample(cands, key), best)

    def test_sampling_frequencies_follow_the_distribution(self):
        s = ContentKeyedSampler(seed=5, m=4)
        cands = {0: math.log(0.6), 1: math.log(0.3), 2: math.log(0.1)}
        n = 20000
        counts = [0, 0, 0]
        for k in range(n):
            counts[s.sample(cands, k)] += 1
        self.assertAlmostEqual(counts[0] / n, 0.6, delta=0.02)
        self.assertAlmostEqual(counts[1] / n, 0.3, delta=0.02)


class TestAlign(unittest.TestCase):
    def test_cursor_rule(self):
        self.assertEqual(align([1, 2, 3, 4], [9, 9], 2, 2), 2)

    def test_longest_suffix_match(self):
        draft = [5, 6, 7, 8, 9, 6, 7, 10]
        self.assertEqual(align(draft, [1, 6, 7], 3, None), 7)      # last match of (6,7) is at 5..6 -> continue at 7
        self.assertEqual(align(draft, [1, 2, 3], 3, None), None)   # no suffix of the tail occurs in the draft

    def test_end_of_draft(self):
        self.assertIsNone(align([1, 2, 3], [2, 3], 2, None))


def _peaked(target, eps):
    def lp(prefix):
        pos = len(prefix)
        if pos >= len(target):
            return {1: 0.0}
        v = target[pos]
        if eps == 0:
            return {v: 0.0}
        return {v: math.log(1 - eps), (v + 1): math.log(eps)}
    return lp


class TestVerify(unittest.TestCase):
    def test_draft_fully_accepted_when_model_unchanged(self):
        target = [11, 12, 13, 14, 15, 1]
        s = ContentKeyedSampler(seed=7, m=3)
        st = KeyState()
        res = verify_chunk(_peaked(target, 0.0), [], target, 0, 64, s, st, eos=(1,))
        self.assertEqual(res.tokens, target)
        self.assertEqual(res.accepted, len(target))
        self.assertEqual(res.decoded, 0)

    def test_first_mismatch_substitutes_keyed_sample_and_realigns(self):
        old = [11, 12, 13, 14, 15, 1]
        new = [11, 12, 99, 14, 15, 1]            # the updated state changes one token in the middle
        s = ContentKeyedSampler(seed=7, m=2)
        st = KeyState()
        res = verify_chunk(_peaked(new, 0.0), [], old, 0, 64, s, st, eos=(1,))
        self.assertEqual(res.tokens, [11, 12, 99]); self.assertEqual(res.accepted, 2); self.assertEqual(res.decoded, 1)
        self.assertEqual(res.draft_cursor, 3)
        # after the divergence the old continuation is tried first (Align rule 1) and accepted again
        res2 = verify_chunk(_peaked(new, 0.0), res.tokens, old, res.draft_cursor, 64, s, st, eos=(1,))
        self.assertEqual(res2.tokens, [14, 15, 1]); self.assertEqual(res2.accepted, 3)

    def test_verification_equals_decoding(self):
        """The accepted prefix is exactly what the keyed sampler decodes on its own (same keys, same noise)."""
        target = [21, 22, 23, 24, 25, 26, 27, 1]
        lp = _peaked(target, 0.25)
        s1, s2 = ContentKeyedSampler(seed=11, m=3), ContentKeyedSampler(seed=11, m=3)
        dec = generate_chunk(lp, [], 64, s1, KeyState(), eos=(1,)).tokens
        ver = verify_chunk(lp, [], dec, 0, 64, s2, KeyState(), eos=(1,))
        self.assertEqual(ver.tokens, dec)
        self.assertEqual(ver.accepted, len(dec))

    def test_content_keys_realign_after_a_length_change(self):
        """A longer tool result shifts positions; content keys reproduce the continuation, position keys do not."""
        rng = random.Random(0)
        cont = [rng.randrange(30, 60) for _ in range(40)] + [1]

        def model(prefix_len_shift):
            def lp(prefix):
                pos = len(prefix) - prefix_len_shift
                if pos < 0 or pos >= len(cont):
                    return {1: 0.0}
                v = cont[pos]
                return {v: math.log(0.7), v + 1: math.log(0.2), v + 2: math.log(0.1)}
            return lp
        for position_keys, expect_high in ((False, True), (True, False)):
            s = ContentKeyedSampler(seed=4, m=4, position_keys=position_keys)
            orig = generate_chunk(model(5), [100] * 5, 64, s, KeyState(), eos=(1,)).tokens
            res = verify_chunk(model(9), [100] * 9, orig, 0, 64, s, KeyState(), eos=(1,))
            if expect_high:
                self.assertGreater(res.accepted, 0.8 * len(orig))
            else:
                self.assertLess(res.accepted, 0.8 * len(orig))


if __name__ == "__main__":
    unittest.main()


class TestEnginePlugin(unittest.TestCase):
    def test_plugin_keys_match_client_keys(self):
        """The engine-side plugin (pure-Python part) and the client derive the same content key for the same window."""
        from nexus.engine.sglang_plugin import content_keyed_sampler as plugin
        rng = random.Random(0)
        for _ in range(500):
            w = [rng.randrange(0, 200000) for _ in range(rng.randrange(1, 9))]
            n = rng.randrange(0, 6)
            self.assertEqual(plugin.window_key(w, n), window_key(w, n))

    def test_plugin_request_key_counts_occurrences(self):
        from nexus.engine.sglang_plugin import content_keyed_sampler as plugin

        class Req:
            def __init__(self, prompt, out):
                self.origin_input_ids, self.output_ids = prompt, out
        k1 = plugin.request_key(Req([1, 2, 3, 4], []), 2)            # window (3,4), first occurrence
        k2 = plugin.request_key(Req([3, 4, 1, 2, 3, 4], []), 2)      # window (3,4) seen once before
        self.assertEqual(k1, window_key([3, 4], 0))
        self.assertEqual(k2, window_key([3, 4], 1))
        r = Req([1, 2, 3, 4], [])
        plugin.request_key(r, 2); r.output_ids.append(7); r.output_ids.append(3); r.output_ids.append(4)
        self.assertEqual(plugin.request_key(r, 2), window_key([3, 4], 1))   # incremental counting across decode steps
