"""Smart Turn: a *decision* (wait | speculate | end), not a silence timeout.
Heuristic by default; plug in an ONNX end-of-turn model (e.g. Pipecat Smart Turn) for a probability."""
from __future__ import annotations
from enum import Enum
import numpy as np

HOLD = {"and", "but", "or", "so", "because", "the", "a", "an", "to", "of", "with", "um", "uh", "like", "if",
        "then", "that", "for", "in", "on", "my", "your", "also", "well", "i", "is", "are"}


class Decision(str, Enum):
    WAIT = "wait"
    SPECULATE = "speculate"
    END = "end"
    DISCARD = "discard"  # click / cough, not speech


class SmartTurn:
    def decide(self, silence_ms, text, voiced_ms, p_complete=None):
        if voiced_ms < 160 and silence_ms >= 320:
            return Decision.DISCARD
        t = text.strip()
        last = t.split()[-1].lower().strip(".,!?") if t else ""
        if p_complete is not None:
            spec, end = (200, 450) if p_complete >= .8 else (700, 1500) if p_complete <= .3 else (350, 800)
        elif not t:
            spec, end = 10 ** 9, 1400
        elif last in HOLD or t.endswith((",", "-", "…")):
            spec, end = 700, 1500
        elif t.endswith((".", "?", "!")):
            spec, end = 200, 500
        else:
            spec, end = 350, 800
        if silence_ms >= end:
            return Decision.END
        if silence_ms >= spec:
            return Decision.SPECULATE
        return Decision.WAIT


class SmartTurnModel:
    """Optional ONNX end-of-turn classifier over the last 8 s of audio (Whisper-style log-mel input)."""

    def __init__(self, path):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.fe = WhisperFeatureExtractor(chunk_length=8)
        self.inp = self.sess.get_inputs()[0].name

    def p_complete(self, audio):
        audio = audio[-16000 * 8:]
        feats = self.fe(audio, sampling_rate=16000, return_tensors="np", padding="max_length",
                        max_length=16000 * 8, truncation=True).input_features.astype(np.float32)
        return float(np.ravel(self.sess.run(None, {self.inp: feats})[0])[0])
