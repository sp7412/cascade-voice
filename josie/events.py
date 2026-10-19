import json, time


class EventBus:
    """Every stage emits here; subscribers render a UI, the JSONL log makes turns replayable."""

    def __init__(self, log_path=None):
        self.subs, self.f = [], (open(log_path, "a") if log_path else None)

    def subscribe(self, fn):
        self.subs.append(fn)

    def emit(self, kind, **data):
        ev = {"t": round(time.time(), 3), "kind": kind, **data}
        if self.f:
            self.f.write(json.dumps(ev, default=str) + "\n")
            self.f.flush()
        for s in self.subs:
            try:
                s(ev)
            except Exception:
                pass
