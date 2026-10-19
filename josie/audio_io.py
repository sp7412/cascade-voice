import numpy as np


class Mic:
    """Pushes 32 ms float32 frames into an asyncio queue from the PortAudio thread."""

    def __init__(self, cfg, loop, queue):
        self.cfg, self.loop, self.q, self.stream = cfg, loop, queue, None

    def start(self):
        import sounddevice as sd

        def cb(indata, frames, t, status):
            try:
                self.loop.call_soon_threadsafe(self._put, indata[:, 0].copy())
            except RuntimeError:
                pass

        self.stream = sd.InputStream(samplerate=self.cfg.sample_rate, channels=1, blocksize=self.cfg.frame,
                                     dtype="float32", callback=cb)
        self.stream.start()

    def _put(self, f):
        if not self.q.full():
            self.q.put_nowait(f)

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


class AEC:
    """Acoustic echo canceller. Uses speexdsp if installed, otherwise passthrough
    (headphones or macOS voice-processing input also work)."""

    def __init__(self, frame=512, sr=16000):
        self.ec = None
        try:
            from speexdsp import EchoCanceller
            self.ec = EchoCanceller.create(frame, frame * 8, sr)
        except Exception:
            pass

    def process(self, mic, ref):
        if self.ec is None or ref is None or len(ref) != len(mic):
            return mic
        near = (np.clip(mic, -1, 1) * 32767).astype(np.int16).tobytes()
        far = (np.clip(ref, -1, 1) * 32767).astype(np.int16).tobytes()
        out = np.frombuffer(self.ec.process(near, far), np.int16)
        return out.astype(np.float32) / 32767
