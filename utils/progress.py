"""
progress.py -- minimal progress logging for long SB-GPAV runs.

One flushed line per event on stderr:
    YYYY-MM-DD HH:MM:SS +<elapsed since start> [<context>] <message>
"""
import datetime
import sys
import time


def _hms(sec: float) -> str:
    sec = int(max(sec, 0))
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return (f"{d}d" if d else "") + f"{h:02d}:{m:02d}:{s:02d}"


class Progress:
    def __init__(self, interval: float = 600.0, enabled: bool = True, stream=None):
        self.interval = float(interval)
        self.enabled = bool(enabled)
        self.stream = stream
        self.ctx = ""                 # e.g. "seg 3/100"; set by the caller
        self.t0 = time.monotonic()

    def log(self, msg: str) -> None:
        if not self.enabled:
            return
        stream = self.stream if self.stream is not None else sys.stderr
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ctx = f"[{self.ctx}] " if self.ctx else ""
        print(f"{stamp} +{_hms(time.monotonic() - self.t0)} {ctx}{msg}",
              file=stream, flush=True)

    def start(self, msg: str) -> float:
        self.log("START " + msg)
        return time.monotonic()

    def end(self, msg: str, t0: float) -> None:
        self.log(f"END   {msg} (took {_hms(time.monotonic() - t0)})")

    def ticker(self, label: str, total: int):
        """Returns None when disabled, so callers guard with `is not None`."""
        return _Ticker(self, label, total) if self.enabled else None


class _Ticker:
    """Logs position/total and rate at most once per `interval` seconds."""

    def __init__(self, p: Progress, label: str, total: int):
        self.p, self.label, self.total = p, label, total
        self.n = 0
        now = time.monotonic()
        self.t0 = self.t_last = now
        self.n_last = 0
        self.next = now + p.interval

    def step(self, pos=None, info=None) -> None:
        if pos is None:
            self.n += 1
        else:
            self.n = pos
        now = time.monotonic()
        if now < self.next:
            return
        dt = now - self.t_last
        rate = (self.n - self.n_last) / dt if dt > 0 else 0.0
        avg = self.n / (now - self.t0) if now > self.t0 else 0.0
        pct = 100.0 * self.n / self.total if self.total else 100.0
        eta = f" naive-ETA={_hms((self.total - self.n) / avg)}" if avg > 0 else ""
        live = f" live_blocks={len(info)}" if info is not None else ""
        self.p.log(f"{self.label}: {self.n}/{self.total} ({pct:.1f}%) "
                   f"rate={rate:.4g}/s (last window) avg={avg:.4g}/s{eta}{live}")
        self.t_last, self.n_last, self.next = now, self.n, now + self.p.interval
