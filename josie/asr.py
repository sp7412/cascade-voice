import platform, tempfile
import numpy as np


class FasterWhisperASR:
    name = "faster-whisper"

    def __init__(self, model="small.en"):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, compute_type="int8")

    def transcribe(self, audio):
        segs, _ = self.m.transcribe(audio, language="en", beam_size=1, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()


class MLXAudioASR:
    """Qwen3-ASR / Parakeet via mlx-audio. These want whole utterances, so the pipeline re-runs them on
    the accumulated audio every 320 ms (partials) and once more at the confirmed end of turn."""
    name = "mlx-audio"

    def __init__(self, model_id):
        from mlx_audio.stt.utils import load_model
        self.m = load_model(model_id)

    def transcribe(self, audio):
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, 16000)
            r = self.m.generate(f.name)
        return (getattr(r, "text", None) or str(r)).strip()


def make_asr(cfg):
    want = cfg.asr_backend
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    if want in ("mlx", "auto") and (apple or want == "mlx"):
        try:
            return MLXAudioASR(cfg.asr_model)
        except Exception as e:
            print(f"[asr] MLX backend failed ({e}); falling back to faster-whisper")
    return FasterWhisperASR(cfg.whisper_model)
