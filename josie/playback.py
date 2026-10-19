"""Player that knows exactly which words the user has heard, supports a 150 ms fade-out,
and a start gate for the RTF-aware pre-roll."""
import asyncio, threading, time
from collections import deque
import numpy as np


class Segment:
    def __init__(self, text, audio):
        self.text, self.audio = text, audio
        words = text.split()
        total = max(1, sum(len(w) + 1 for w in words))
        acc, self.words = 0, []
        for w in words:  # word end-times estimated proportionally to character count
            acc += len(w) + 1
            self.words.append((w, int(len(audio) * acc / total)))


class Player:
    def __init__(self, sample_rate=24000, device=None, ref_seconds=2.0):
        self.sr, self.device = sample_rate, device
        self.q, self.done, self.cur, self.pos = deque(), [], None, 0
        self.lock = threading.Lock()
        self.start_at = 0.0
        self.fade_total = self.fade_left = 0
        self.stopped = False
        self.started = asyncio.Event()
        self.ref = deque(maxlen=int(ref_seconds * 10))  # recent output blocks for AEC / echo check
        self.stream = None

    def open(self):
        import sounddevice as sd
        self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32", device=self.device,
                                      callback=self._cb)
        self.stream.start()

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    # ---- control (called from the event loop)
    def arm(self, delay):
        self.start_at = time.monotonic() + max(0.0, delay)

    def enqueue(self, text, audio):
        with self.lock:
            if self.stopped:
                return
            self.q.append(Segment(text, audio.astype(np.float32)))

    def fade_out(self, ms=150):
        with self.lock:
            self.fade_total = self.fade_left = max(1, int(self.sr * ms / 1000))

    def flush(self):
        with self.lock:
            self.q.clear(); self.cur = None; self.pos = 0
            self.fade_total = self.fade_left = 0
            self.stopped = False
            self.done = []
        self.started = asyncio.Event()

    @property
    def active(self):
        with self.lock:
            return (self.cur is not None or bool(self.q)) and not self.stopped

    async def wait_idle(self):
        while self.active:
            await asyncio.sleep(0.02)

    async def wait_started(self, timeout=1.5):
        try:
            await asyncio.wait_for(self.started.wait(), timeout)
        except asyncio.TimeoutError:
            pass

    def spoken_text(self):
        """Exactly the words the listener heard, up to the playback cursor."""
        with self.lock:
            parts = [s.text for s in self.done]
            if self.cur is not None:
                heard = [w for w, end in self.cur.words if end <= self.pos]
                if heard:
                    parts.append(" ".join(heard))
        return " ".join(parts).strip()

    def reference(self, n_samples, out_sr=16000):
        """Last n_samples of what we played, resampled to out_sr, for AEC / echo correlation."""
        with self.lock:
            blocks = list(self.ref)
        if not blocks:
            return None
        x = np.concatenate(blocks)
        need = int(n_samples * self.sr / out_sr)
        x = x[-need:]
        if len(x) < need:
            x = np.pad(x, (need - len(x), 0))
        return np.interp(np.linspace(0, len(x) - 1, n_samples), np.arange(len(x)), x).astype(np.float32)

    # ---- audio thread
    def _cb(self, out, frames, t, status):
        out[:] = 0
        with self.lock:
            if self.stopped or time.monotonic() < self.start_at:
                return
            n = 0
            while n < frames:
                if self.cur is None:
                    if not self.q:
                        break
                    self.cur, self.pos = self.q.popleft(), 0
                    if not self.started.is_set():
                        self._signal_started()
                seg = self.cur
                take = min(frames - n, len(seg.audio) - self.pos)
                out[n:n + take, 0] = seg.audio[self.pos:self.pos + take]
                self.pos += take
                n += take
                if self.pos >= len(seg.audio):
                    self.done.append(seg)
                    self.cur = None
            if self.fade_total:
                idx = np.arange(frames)
                gain = np.clip((self.fade_left - idx) / self.fade_total, 0, 1).astype(np.float32)
                out[:, 0] *= gain
                self.fade_left -= frames
                if self.fade_left <= 0:
                    self.stopped = True
            self.ref.append(out[:, 0].copy())

    def _signal_started(self):
        try:
            loop = self._loop
            loop.call_soon_threadsafe(self.started.set)
        except AttributeError:
            pass

    def bind_loop(self, loop):
        self._loop = loop
