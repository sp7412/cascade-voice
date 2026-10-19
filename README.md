# Cascade Voice / JOSIE

A modular, cascaded real-time voice-agent prototype. The intended pipeline is:

```text
microphone → echo/VAD/turn detection → streaming ASR → LLM + tools
           → sentence chunking → streaming TTS → playback/interruption
```

The project aims to keep turn-taking responsive while preserving safe tool semantics:
partial transcripts may trigger speculative text generation, but side-effecting tools
must wait for a confirmed final transcript. Conversation history should reflect audio
that was actually played, not text that was generated but interrupted.

## Status

This repository is being bootstrapped from the `josie-voice.zip` project archive.
Treat this as an experimental prototype, not a production-ready voice service.
Model identifiers, optional backend APIs, audio-device behavior, and end-to-end
latency must be verified on the target hardware before relying on them.

## Whitepaper\n\nThe engineering whitepaper is in [`docs/whitepaper/`](docs/whitepaper/README.md). It covers architecture, turn lifecycle, security boundaries, failure diagnosis, evaluation metrics, and production-readiness gates.\n\n## Development

Python 3.10+ is declared by the package. A local install may require platform-specific
audio and ML dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

For macOS/Apple Silicon, optional MLX backends may be installed separately. See the
project configuration and CLI help for supported extras and model setup.

## Quality and safety checks

- Never execute tools from speculative/partial transcripts.
- Bound tool execution with deadlines and cancellation.
- Treat model output and tool arguments as untrusted input.
- Keep API credentials in environment variables; do not commit secrets.
- Bind any HTTP service to localhost by default. Before exposing it to a network,
  add authentication, request-size limits, rate limits, and TLS at the edge.
- Validate real-time audio behavior with headphones and test echo/barge-in scenarios.
- Measure turn-end-to-first-audio latency, interruption latency, tool timeout behavior,
  and transcript/playback consistency on target hardware.

## Tests

The unit tests cover core text chunking, think-tag stripping, turn decisions, playback
history, and tool timeout/cancellation. They do not validate real microphones,
speaker echo cancellation, downloaded model weights, or real-time performance.

## Deployment

The Dockerfile is intended for the HTTP API path, not a full host microphone/speaker
runtime. Review dependencies and security before deployment. Do not expose the
unauthenticated API directly to the public internet.

## License

No license has been selected yet. Until one is added, default copyright restrictions
apply; do not assume the code is available for reuse.
