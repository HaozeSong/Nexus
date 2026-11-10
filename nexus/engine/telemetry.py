"""Engine telemetry: sample SGLang/vLLM Prometheus /metrics in the background (engine load for the waiting policy, GPU-second attribution)."""
from __future__ import annotations

import re
import threading
import time
import urllib.request

_KEYS = {"running": r"sglang:num_running_reqs\S*\s+([0-9.e+-]+)", "queue": r"sglang:num_queue_reqs\S*\s+([0-9.e+-]+)",
         "token_usage": r"sglang:token_usage\S*\s+([0-9.e+-]+)", "cache_hit": r"sglang:cache_hit_rate\S*\s+([0-9.e+-]+)"}


class MetricsSampler(threading.Thread):
    def __init__(self, root_url: str, interval: float = 1.0):
        super().__init__(daemon=True)
        self.url = root_url.rstrip("/") + "/metrics"
        self.interval = interval
        self.samples: list[dict] = []
        self.lock = threading.Lock()
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                txt = urllib.request.urlopen(self.url, timeout=3).read().decode()
                s = {"t": time.time()}
                for k, pat in _KEYS.items():
                    m = re.findall(pat, txt)
                    s[k] = float(m[-1]) if m else None
                with self.lock:
                    self.samples.append(s)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def latest(self) -> dict:
        with self.lock:
            return dict(self.samples[-1]) if self.samples else {}

    def gpu_seconds(self, t0: float, t1: float) -> float:
        """Share of engine time attributable to one request active in [t0, t1]: integral of 1/max(1, running)."""
        with self.lock:
            pts = [s for s in self.samples if t0 - self.interval <= s["t"] <= t1 + self.interval and s.get("running") is not None]
        if not pts:
            return t1 - t0
        total = 0.0
        prev_t = t0
        for s in pts:
            t = min(max(s["t"], t0), t1)
            total += (t - prev_t) / max(1.0, s["running"])
            prev_t = t
        total += (t1 - prev_t) / max(1.0, pts[-1]["running"])
        return total
