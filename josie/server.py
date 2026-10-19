"""OpenAI-compatible /v1/audio/transcriptions and /v1/audio/speech that stream even though the
underlying models do not: ASR re-runs on growing prefixes (SSE deltas); TTS streams per sentence."""
import io, json, struct
import numpy as np
from .asr import make_asr
from .chunker import SentenceChunker, clean_for_tts
from .config import Config
from .executor import SerialExecutor
from .tts import make_tts


def wav_header(sr):  # streaming WAV: unknown length
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


def pcm16(a):
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


def split_sentences(text):
    c = SentenceChunker()
    return c.feed(text + " ") + c.flush()


def create_app(cfg=None):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse, StreamingResponse
    cfg = cfg or Config()
    app, ex = FastAPI(title="josie-voice"), SerialExecutor()
    asr, tts = make_asr(cfg), make_tts(cfg)

    def load(raw):
        import soundfile as sf
        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        x = x.mean(1)
        return x if sr == 16000 else np.interp(np.linspace(0, len(x) - 1, int(len(x) * 16000 / sr)),
                                               np.arange(len(x)), x).astype(np.float32)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...), model: str = Form("josie"), stream: bool = Form(False)):
        audio = load(await file.read())
        if not stream:
            return {"text": await ex.run(asr.transcribe, audio)}

        async def gen():
            prev, step = "", 16000 * 3 // 2
            for end in list(range(step, len(audio), step)) + [len(audio)]:
                text = await ex.run(asr.transcribe, audio[:end])
                if end == len(audio):
                    yield f"data: {json.dumps({'type': 'transcript.text.done', 'text': text})}\n\n"
                elif text.startswith(prev) and len(text) > len(prev):
                    yield f"data: {json.dumps({'type': 'transcript.text.delta', 'delta': text[len(prev):]})}\n\n"
                    prev = text
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/audio/speech")
    async def speech(body: dict):
        text, fmt = clean_for_tts(body.get("input", "")), body.get("response_format", "pcm")
        if not text:
            return JSONResponse({"error": "empty input"}, status_code=400)

        async def gen():
            if fmt == "wav":
                yield wav_header(tts.sample_rate)
            for s in split_sentences(text):
                yield pcm16(await ex.run(tts.synth, s))
        return StreamingResponse(gen(), media_type="audio/wav" if fmt == "wav" else "audio/L16",
                                 headers={"X-Sample-Rate": str(tts.sample_rate)})

    @app.get("/health")
    async def health():
        return {"asr": asr.name if hasattr(asr, "name") else "?", "tts": tts.name}

    return app
