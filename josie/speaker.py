"""Assistant-voice detection so barge-in ignores our own speech.
Primary: WeSpeaker ONNX embeddings (cosine vs. running centroid of our TTS output).
Fallback: normalized cross-correlation between mic audio and what we just played."""
import numpy as np


def _rs(x, sr_from, sr_to=16000):
    if sr_from == sr_to or len(x) == 0:
        return x.astype(np.float32)
    n = int(len(x) * sr_to / sr_from)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def echo_correlation(mic, ref):
    """Peak normalized cross-correlation (FFT) of mic vs reference; ~1 means mic is mostly our own output."""
    if ref is None or len(mic) == 0 or len(ref) == 0:
        return 0.0
    n = 1 << (len(mic) + len(ref)).bit_length()
    a, b = mic - mic.mean(), ref - ref.mean()
    cc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.max(np.abs(cc)) / denom)


class SpeakerEmbedder:
    def __init__(self, onnx_path=""):
        self.ok = False
        if not onnx_path:
            return
        try:
            import onnxruntime as ort
            import torchaudio  # noqa: F401  (fbank features)
            self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.ok = True
        except Exception as e:
            print(f"[speaker] embedder unavailable ({e}); using echo-correlation fallback")

    def embed(self, audio16k):
        import torch, torchaudio.compliance.kaldi as K
        w = torch.from_numpy(audio16k).unsqueeze(0) * 32768
        f = K.fbank(w, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=16000)
        f = (f - f.mean(0, keepdim=True)).unsqueeze(0).numpy().astype(np.float32)
        e = self.sess.run(None, {self.sess.get_inputs()[0].name: f})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)


class AssistantVoice:
    def __init__(self, embedder: SpeakerEmbedder, thr=0.6):
        self.emb, self.thr, self.centroid, self.n = embedder, thr, None, 0

    def observe(self, audio, sr):
        """Called (off the event loop) with each synthesized sentence."""
        if not self.emb.ok or len(audio) < sr // 2:
            return
        e = self.emb.embed(_rs(audio, sr))
        self.centroid = e if self.centroid is None else (self.centroid * self.n + e) / (self.n + 1)
        self.n += 1

    def is_assistant(self, mic16k, ref16k=None):
        if self.emb.ok and self.centroid is not None:
            e = self.emb.embed(mic16k)
            c = self.centroid / (np.linalg.norm(self.centroid) + 1e-9)
            return float(e @ c) >= self.thr
        return echo_correlation(mic16k, ref16k) >= 0.5
