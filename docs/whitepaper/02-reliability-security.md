# 2. Reliability, security, and failure handling

## 2.1 Threat model

Assume that spoken input, ASR transcripts, LLM output, tool arguments, web results,
and files read by tools can all be incorrect or adversarial. A model response is not
a security boundary. The application must enforce what the model is permitted to do.

| Boundary | Failure or threat | Required control |
|---|---|---|
| Audio upload | Oversized or malformed media exhausts memory/CPU | Byte limits, decode validation, request timeouts |
| HTTP API | Unauthenticated remote use, cost abuse, data exposure | Bearer auth, network restriction, TLS, rate and concurrency limits |
| LLM stream | Invalid JSON, unexpected tool, provider outage | Strict parsing, allow-list, bounded retries, graceful fallback |
| Tool arguments | Wrong types, extra fields, prompt injection | Schema validation, per-tool authorization, argument limits |
| Tool process | Hanging subprocess, child-process leak | Deadline, process-group termination, await process exit |
| Workspace | Traversal or symlink escape | Canonical path checks, least-privilege filesystem access |
| Cancellation | Stale tool/LLM/TTS work survives barge-in | Cancellation propagation and cleanup tests |
| Playback/history | History records words not heard | Playback cursor accounting and interruption regression tests |
| Logs | Transcripts and arguments leak | Redaction, restricted permissions, retention policy |
| Deployment | Prototype accidentally exposed publicly | Safe bind defaults, required secret, TLS/rate limit at edge |

## 2.2 HTTP API safeguards

The HTTP API is separate from the native microphone/speaker loop. It provides
OpenAI-style audio transcription and speech routes, but compatibility is intentionally
limited: the implementation supports PCM/WAV output and its own streaming transcript
event shape, not every OpenAI API option.

The service should remain loopback-only for development. When binding to a non-loopback
interface, set a high-entropy `JOSIE_API_KEY` and put the service behind TLS and a
reverse proxy with request-rate and concurrency limits. A bearer key is not a substitute
for transport security. Rotate it if exposed; do not put it in source control, URLs,
logs, or screenshots.

The application enforces a 25 MiB audio-read limit and a 20,000-character speech-input
limit. These are defense-in-depth controls, not a complete denial-of-service defense:
the reverse proxy should also limit total request size, connection duration, request
rate, concurrent inference, and total response time. A production API should add
structured error responses, metrics, dependency readiness checks, and per-caller quotas.

## 2.3 Tool execution is a privilege boundary

The built-in Python runner is disabled by default. Setting
`JOSIE_ENABLE_PYTHON_TOOL=1` enables code execution with the privileges of the JOSIE
process. This opt-in is only a warning gate, not a sandbox; it remains intended for a
trusted local prototype only. It must not be enabled in a
multi-user or remotely accessible service without an operating-system sandbox,
resource limits, filesystem/network policy, and a separate low-privilege identity.

Path canonicalization helps prevent traversal, but it does not make the workspace a
complete sandbox. Filesystem permissions, symlink races, hard links, process access,
and inherited environment variables remain relevant. Keep the service account's
permissions narrow and do not point the workspace at a home directory or secrets tree.

Tool schemas should reject unknown properties and enforce string, list, and payload
length limits. Each operation should have an explicit risk tier:

- **Tier 0 — read-only:** local time, safe status checks.
- **Tier 1 — bounded retrieval:** web search, scoped file reads.
- **Tier 2 — local mutation:** file writes and application controls.
- **Tier 3 — arbitrary execution/external side effects:** Python execution, sending
  messages, bookings, payments, or remote writes.

Higher tiers need stronger policy checks, user confirmation where appropriate,
idempotency, and audit trails. Tool output must be treated as untrusted context,
not as new instructions.

## 2.4 Cancellation and cleanup

Cancellation must stop work, not merely stop waiting for it. For every asynchronous
stage, define what cancellation means and verify it:

- LLM HTTP streams close their response context when their task is cancelled.
- Tool tasks are cancelled and awaited; subprocesses are terminated and reaped.
- TTS workers are cancelled and awaited, and their queues cannot leak into the next turn.
- Playback queues and current segments are flushed at the correct point in the fade.
- Background agents have an explicit lifetime, concurrency cap, and shutdown policy.
- A cancelled operation must not later publish a stale result or mutate the new turn.

Cancellation cannot roll back a completed external side effect. For state-changing
tools, use idempotency keys and operation-status checks, and design compensation or
manual recovery for ambiguous outcomes.

## 2.5 Failure taxonomy

| Symptom | Evidence to collect | Likely causes | Corrective action |
|---|---|---|---|
| Assistant speaks over caller | Interruption timestamp, player state, audio callback trace | Slow VAD, echo classified as near-end speech, fade not applied | Tune VAD/AEC, measure interrupt-to-silence, test with speakers and headphones |
| Assistant stops but stale answer resumes | Turn ID, task lifecycle, tool completion timestamps | Unawaited task, old queue, missing cancellation propagation | Add turn-scoped cancellation and cleanup; reject stale turn IDs |
| Tool action happens on a wrong phrase | Partial/final transcript pair, tool dispatch event | Speculative side effect or missing confirmation | Gate tools on final transcript; require confirmation for high-risk actions |
| Transcript SSE repeats or omits words | ASR prefix outputs and emitted deltas | ASR revises a prior prefix rather than appending | Use replaceable partial events or diff-based revision semantics; final event is authoritative |
| Long pause before speech | Stage timings, TTS RTF, queue depth | Cold model load, slow first synthesis, excessive pre-roll | Warm models, cap queued text, measure first-audio latency by backend |
| API memory spikes | Request bytes, decoded duration, concurrency | Unbounded upload or concurrent decode | Enforce byte/duration limits and semaphore; add proxy-level request limits |
| History includes unspoken text | Playback cursor, segment boundaries, interruption point | Estimated timing, race between flush and commit | Track exact sample offsets and add deterministic barge-in tests |
| Search or model request hangs | Request timing, cancellation reason | Missing timeout or network stall | Set connect/read/write/pool timeouts and bound retries |

Use the diagnostic sequence **symptom → evidence → root cause → corrective action → regression test**.
Avoid fixing a symptom based only on a single anecdotal trace.

## 2.6 Logging and privacy

The JSONL event bus can record transcript text, tool arguments, tool results, and assistant
speech. Treat logs as sensitive data. Production logging should use structured event
schemas, omit secrets, redact personal data, set file permissions, define retention and
deletion, and separate operational metrics from conversation content. Do not log bearer
tokens or full audio payloads.

## 2.7 Current limitations

- The current test suite is small and does not exercise actual audio devices or models.
- AEC may be pass-through when its optional dependency is absent.
- Barge-in identity detection can fall back to echo correlation; this is not robust
  speaker authentication.
- Playback word timing is estimated, not aligned to exact acoustic output.
- Built-in Python execution is not sandboxed.
- The HTTP API is not a complete OpenAI API implementation and needs deployment controls.
- Model IDs, MLX APIs, device support, and performance must be checked against installed
  versions on the intended target.

Do not market the prototype as secure, real-time guaranteed, or production-ready until
the acceptance criteria in [Chapter 3](03-evaluation.md) have passed.
