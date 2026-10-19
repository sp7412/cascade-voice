import platform
import numpy as np


class KokoroTTS:
    name, sample_rate = "kokoro", 24000

    def __init__(self, voice="af_heart"):
        from kokoro import KPipeline
        self.p, self.voice = KPipeline(lang_code="a"), voice

    def synth(self, text):
        parts = [np.asarray(a, np.float32).reshape(-1) for _, _, a in self.p(text, voice=self.voice)]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)


class MLXAudioTTS:
    name = "mlx-audio"

    def __init__(self, model_id, voice=None):
        from mlx_audio.tts.utils import load_model
        self.m, self.voice = load_model(model_id), voice
        self.sample_rate = getattr(self.m, "sample_rate", 24000)

    def synth(self, text):
        kw = {"voice": self.voice} if self.voice else {}
        parts = [np.array(r.audio, dtype=np.float32).reshape(-1) for r in self.m.generate(text=text, **kw)]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def make_tts(cfg):
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    if cfg.tts_backend in ("mlx", "auto") and (apple or cfg.tts_backend == "mlx"):
        try:
            return MLXAudioTTS(cfg.tts_model, cfg.persona.voice)
        except Exception as e:
            print(f"[tts] MLX backend failed ({e}); falling back to Kokoro")
    return KokoroTTS(cfg.kokoro_voice)


class RTFTracker:
    """EMA of synthesis speed s = audio_seconds / wall_seconds (s>1 means faster than real time).
    preroll(D) = D*(1-s)*1.1 + 0.15, capped at 3 s; zero extra delay when s >= 1.05."""

    def __init__(self, alpha=0.3):
        self.alpha, self.speed = alpha, None

    def update(self, audio_s, wall_s):
        if wall_s <= 0 or audio_s <= 0:
            return
        s = audio_s / wall_s
        self.speed = s if self.speed is None else self.alpha * s + (1 - self.alpha) * self.speed

    def preroll(self, d):
        s = self.speed
        if s is None:
            return 0.3  # cold start guess
        if s >= 1.05:
            return 0.0
        return min(3.0, max(0.0, d * (1 - s) * 1.1 + 0.15))
