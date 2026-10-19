"""OpenAI-compatible audio endpoints for the JOSIE prototype.

The HTTP API is intentionally separate from the host microphone/speaker loop.
Set JOSIE_API_KEY before binding this service to a non-loopback interface.
"""
import hmac
import io
import json
import os
import struct

import numpy as np

from .asr import make_asr
from .chunker import SentenceChunker, clean_for_tts
from .config import Config
from .executor import SerialExecutor
from .tts import make_tts

MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_SPEECH_CHARS = 20_000


def authorized(api_key: str, authorization: str | None) -> bool:
    """Check a bearer token without leaking timing information."""
    if not api_key:
        return True  # Local-only mode; CLI blocks unkeyed non-loopback binds.
    if not authorization or not authorization.startswith("Bearer "):
        return False
    supplied = authorization[7:]
    return hmac.compare_digest(supplied, api_key)


def wav_header(sr):
    # Streaming WAV: total length is unknown until the stream finishes.
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


def pcm16(a):
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


def split_sentences(text):
    c = SentenceChunker()
    return c.feed(text + " ") + c.flush()


def create_app(cfg=None):
    from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, StreamingResponse

    cfg = cfg or Config()
    app, ex = FastAPI(title="josie-voice"), SerialExecutor()
    asr, tts = make_asr(cfg), make_tts(cfg)
    api_key = os.getenv("JOSIE_API_KEY", "")

    async def check_auth(authorization: str | None = Header(default=None)):
        if not authorized(api_key, authorization):
            raise HTTPException(status_code=401, detail="unauthorized",
                                headers={"WWW-Authenticate": "Bearer"})

    def load(raw):
        import soundfile as sf

        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        if not len(x):
            raise ValueError("audio file is empty")
        x = x.mean(1)
        if sr != 16000:
            target_len = max(1, int(len(x) * 16000 / sr))
            x = np.interp(
                np.linspace(0, len(x) - 1, target_len), np.arange(len(x)), x
            ).astype(np.float32)
        return x

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(check_auth)])
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form("josie"),
        stream: bool = Form(False),
    ):
        raw = await file.read(MAX_AUDIO_BYTES + 1)
        await file.close()
        if len(raw) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="audio exceeds 25 MiB limit")
        if not raw:
            raise HTTPException(status_code=400, detail="audio file is empty")
        try:
            audio = load(raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid or unsupported audio file") from exc

        if not stream:
            return {"text": await ex.run(asr.transcribe, audio)}

        async def gen():
            prev, step = "", 16000 * 3 // 2
            for end in list(range(step, len(audio), step)) + [len(audio)]:
                text = await ex.run(asr.transcribe, audio[:end])
                if end == len(audio):
                    yield f"data: {json.dumps({'type': 'transcript.text.done', 'text': text})}\\n\\n"
                elif text.startswith(prev) and len(text) > len(prev):
                    yield f"data: {json.dumps({'type': 'transcript.text.delta', 'delta': text[len(prev):]})}\\n\\n"
                    prev = text
            yield "data: [DONE]\\n\\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.post("/v1/audio/speech", dependencies=[Depends(check_auth)])
    async def speech(body: dict):
        text, fmt = clean_for_tts(body.get("input", "")), body.get("response_format", "pcm")
        if not text:
            return JSONResponse({"error": "empty input"}, status_code=400)
        if len(text) > MAX_SPEECH_CHARS:
            raise HTTPException(status_code=413, detail="input exceeds 20000 characters")
        if fmt not in {"pcm", "wav"}:
            raise HTTPException(status_code=400, detail="response_format must be pcm or wav")

        async def gen():
            if fmt == "wav":
                yield wav_header(tts.sample_rate)
            for sentence in split_sentences(text):
                yield pcm16(await ex.run(tts.synth, sentence))

        media_type = "audio/wav" if fmt == "wav" else "audio/pcm"
        return StreamingResponse(
            gen(), media_type=media_type,
            headers={"X-Sample-Rate": str(tts.sample_rate),
                     "Cache-Control": "no-store"},
        )

    @app.get("/health", dependencies=[Depends(check_auth)])
    async def health():
        return {"asr": getattr(asr, "name", "?"), "tts": tts.name}

    return app
