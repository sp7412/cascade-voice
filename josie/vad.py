import os
import numpy as np


class SileroVAD:
    """Silero VAD on ONNX Runtime (CPU), 512-sample (32 ms) windows @16 kHz."""

    def __init__(self, path=None):
        import onnxruntime as ort
        if path is None:
            import silero_vad
            path = os.path.join(os.path.dirname(silero_vad.__file__), "data", "silero_vad.onnx")
        so = ort.SessionOptions()
        so.inter_op_num_threads = so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros((1, 64), np.float32)

    def __call__(self, frame):
        x = np.concatenate([self.ctx, frame[None].astype(np.float32)], 1)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": np.array(16000, np.int64)})
        self.ctx = x[:, -64:]
        return float(out[0][0])


class EnergyVAD:
    """Last-resort fallback so the pipeline still runs without the ONNX model."""

    def __call__(self, frame):
        return min(1.0, float(np.sqrt(np.mean(frame ** 2))) * 40)

    def reset(self):
        pass


def load_vad():
    try:
        return SileroVAD()
    except Exception as e:
        print(f"[vad] Silero unavailable ({e}); using energy VAD")
        return EnergyVAD()
