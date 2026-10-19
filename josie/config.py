from __future__ import annotations
import os, random
from dataclasses import dataclass, field


def _e(k, d):
    return os.getenv(k, d)


@dataclass
class Persona:
    name: str = "Josie"
    system: str = (
        "You are Josie, a voice assistant in a live spoken call. Rules: speak in short, natural "
        "sentences, never use markdown, lists, emoji or URLs. Before calling any tool, say one short "
        "sentence about what you are about to do (e.g. 'Let me check your calendar.'). Call independent "
        "tools in parallel. After tools return, summarize the result in one or two sentences. If a task "
        "is long, call spawn_agent and keep chatting. Write numbers and times the way you would say them aloud."
    )
    fillers: tuple = ("One sec.", "Let me look.", "Okay, checking.", "Sure, give me a moment.")
    asides: tuple = ("Still working on it.", "Almost there.", "Bear with me, this one's a bit slow.")
    voice: str | None = None

    def filler(self) -> str:
        return random.choice(self.fillers)

    def aside(self) -> str:
        return random.choice(self.asides)


PERSONAS = {"josie": Persona(), "terse": Persona(name="Terse", system=Persona.system + " Be extremely brief.")}


@dataclass
class Config:
    # audio
    sample_rate: int = 16000
    frame: int = 512  # 32 ms @ 16 kHz (Silero VAD window)
    vad_threshold: float = 0.5
    echo_gate_boost: float = 0.25  # added to VAD threshold while assistant is speaking
    partial_interval_ms: int = 320
    # smart turn
    smart_turn_onnx: str = _e("JOSIE_SMART_TURN_ONNX", "")  # optional model file; heuristic if empty
    # ASR
    asr_backend: str = _e("JOSIE_ASR", "auto")  # auto | mlx | faster-whisper
    asr_model: str = _e("JOSIE_ASR_MODEL", "mlx-community/Qwen3-ASR-0.6B-4bit")  # verify repo id on HF
    whisper_model: str = _e("JOSIE_WHISPER_MODEL", "small.en")
    # LLM (Ollama OpenAI-compatible; cloud: base=https://ollama.com + key)
    llm_base: str = _e("JOSIE_LLM_BASE", "http://localhost:11434")
    llm_key: str = _e("OLLAMA_API_KEY", "")
    llm_model: str = _e("JOSIE_LLM_MODEL", "qwen3:8b")  # set to your exact DeepSeek Flash tag for the demo config
    temperature: float = 0.6
    max_tool_rounds: int = 5
    tool_deadline_s: float = 15.0
    progress_every_s: float = 12.0
    max_asides: int = 3
    tool_waits_for_preamble: bool = True
    bg_announce_quiet_s: float = 1.2
    max_history_chars: int = 24000
    # TTS
    tts_backend: str = _e("JOSIE_TTS", "auto")  # auto | mlx | kokoro
    tts_model: str = _e("JOSIE_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit")  # verify repo id on HF
    kokoro_voice: str = _e("JOSIE_KOKORO_VOICE", "af_heart")
    # barge-in
    bargein_ms: int = 1000
    fade_ms: int = 150
    speaker_onnx: str = _e("JOSIE_SPEAKER_ONNX", "")  # WeSpeaker ONNX; falls back to echo correlation
    speaker_threshold: float = 0.6
    # misc
    workspace: str = _e("JOSIE_WORKSPACE", "workspace")
    log_dir: str = "logs"
    persona: Persona = field(default_factory=Persona)
