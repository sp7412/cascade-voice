import argparse, asyncio, os, sys, time
from .config import Config, PERSONAS
from .events import EventBus

C = {"dim": "\033[2m", "b": "\033[1m", "g": "\033[32m", "y": "\033[33m", "c": "\033[36m", "r": "\033[31m", "0": "\033[0m"}
DEMO_SCRIPT = """Suggested 6-minute flow (talk naturally, interrupt whenever you like):
  1. "What time is it, and what's on my calendar today?"        -> get_time + calendar_today in parallel
  2. "Search for the latest Apple Silicon news and summarize."   -> web_search with spoken preamble
  3. Interrupt mid-answer: "Actually, skip that."                -> barge-in, history cut at what you heard
  4. "Write a Python script that prints primes under 100, save it as primes.py and run it."  -> write_file + run_python
  5. "In the background, research three tips for fast TTS and tell me when done." -> spawn_agent, keep chatting
  6. Keep asking small tool questions; after ~100 calls the history still stays bounded."""


def render(ev):
    k, t = ev["kind"], time.strftime("%H:%M:%S", time.localtime(ev["t"]))
    show = {
        "partial": f"{C['dim']}  … {ev.get('text','')}{C['0']}",
        "final": f"{C['b']}YOU   {ev.get('text','')}{C['0']}",
        "assistant_sentence": f"{C['g']}JOSIE {ev.get('text','')}{C['0']}",
        "tool_start": f"{C['y']}  ⚙ {ev.get('name')} {ev.get('args')}{C['0']}",
        "tool_end": f"{C['y']}  ✓ {ev.get('name')} {ev.get('ms')}ms{C['0']}",
        "speculate": f"{C['c']}  ⚡ speculating on: {ev.get('text')}{C['0']}",
        "speculation_hit": f"{C['c']}  ⚡ speculation HIT{C['0']}",
        "speculation_miss": f"{C['c']}  ⚡ speculation miss{C['0']}",
        "interrupt": f"{C['r']}  ✋ barge-in{C['0']}",
        "bg_started": f"{C['c']}  ↗ background agent {ev.get('n')}: {ev.get('task')}{C['0']}",
        "bg_done": f"{C['c']}  ↙ background agent {ev.get('n')} done{C['0']}",
        "tts": f"{C['dim']}  tts {ev.get('audio_s')}s audio in {ev.get('wall_s')}s (speed {ev.get('speed')}x){C['0']}",
        "error": f"{C['r']}  ERROR {ev}{C['0']}",
    }.get(k)
    if show:
        print(f"{C['dim']}{t}{C['0']} {show}", flush=True)


def build(args, audio=True):
    from .pipeline import Voice
    cfg = Config(persona=PERSONAS[args.persona])
    os.makedirs(cfg.log_dir, exist_ok=True)
    bus = EventBus(os.path.join(cfg.log_dir, f"session-{int(time.time())}.jsonl"))
    bus.subscribe(render)
    return Voice(cfg, bus, audio=audio)


async def amain(args):
    if args.cmd == "text":
        v = build(args, audio=not args.mute)
        while True:
            line = await asyncio.get_running_loop().run_in_executor(None, input, "you> ")
            await v.run_text(line)
    v = build(args)
    if args.cmd == "demo":
        print(f"{C['b']}JOSIE-Voice demo{C['0']}\n{DEMO_SCRIPT}\n")
    print("Listening. Use headphones unless AEC/speaker-ID is configured. Ctrl-C to quit.")
    await v.run()


def models():
    c = Config()
    print(f"""# Models (verify exact repo ids on Hugging Face / Ollama before downloading)
ollama pull {c.llm_model}                      # or: ollama signin  (cloud tags run remotely)
huggingface-cli download {c.asr_model}
huggingface-cli download {c.tts_model}
pip install silero-vad                          # ships silero_vad.onnx
huggingface-cli download Wespeaker/wespeaker-voxceleb-resnet34-LM   # -> set JOSIE_SPEAKER_ONNX to the .onnx
huggingface-cli download pipecat-ai/smart-turn-v3                   # optional -> JOSIE_SMART_TURN_ONNX""")


def main():
    ap = argparse.ArgumentParser("josie")
    ap.add_argument("cmd", choices=["run", "demo", "text", "serve", "models"])
    ap.add_argument("--persona", default="josie", choices=list(PERSONAS))
    ap.add_argument("--mute", action="store_true", help="text mode: no audio output")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    if a.cmd == "models":
        return models()
    if a.cmd == "serve":
        import ipaddress
        host = a.host.strip("[]")
        is_loopback = host.lower() == "localhost"
        try:
            is_loopback = is_loopback or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        if not is_loopback and not os.getenv("JOSIE_API_KEY"):
            raise SystemExit(
                "Refusing non-loopback bind without JOSIE_API_KEY. Set a strong key first."
            )
        import uvicorn
        from .server import create_app
        return uvicorn.run(create_app(), host=a.host, port=a.port)
    try:
        asyncio.run(amain(a))
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)


if __name__ == "__main__":
    main()
