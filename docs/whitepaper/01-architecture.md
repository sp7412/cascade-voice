# 1. Architecture and turn lifecycle

## 1.1 Design intent

JOSIE separates the real-time voice loop into replaceable stages:

```text
Microphone
   │  fixed-size PCM frames
   ▼
AEC / echo gate ──► VAD ──► turn detector
                              │
                     partial ASR / speculation
                              │ final transcript
                              ▼
                        LLM orchestrator
                         │           ▲
                    tool calls    tool results
                         │           ▲
                         └─ bounded tool registry
                              │ assistant text
                              ▼
                       think-tag filter
                              │
                       sentence chunker
                              │
                         streaming TTS
                              │ audio segments
                              ▼
                      playback / barge-in
                              │
                   played-text history commit
```

The diagram is conceptual, not a claim that every stage currently has a production-grade
implementation. In particular, the AEC class falls back to pass-through when its
optional dependency is unavailable, and the turn detector can use a heuristic when
the optional Smart Turn model is not configured.

## 1.2 Components and responsibilities

| Component | Responsibility | Key contract |
|---|---|---|
| Audio input | Capture mono float PCM frames | Queue pressure must not block the audio callback |
| AEC / echo gate | Reduce playback leakage into recognition | Pass-through fallback must be visible to operators |
| VAD and turn detector | Detect speech regions and likely end of turn | A turn decision is a hypothesis, not a transcript |
| ASR | Produce partial and final text | Only final text is authoritative for side effects |
| Speculation | Start LLM generation from a partial transcript | Speculative output is reusable only if final text matches |
| LLM adapter | Stream text and assemble tool calls | Provider output is untrusted and may be malformed |
| Tool registry | Validate known operations and execute with deadlines | Tools must be bounded, observable, and cancellable |
| Sentence chunker | Release speakable clauses/sentences early | Avoid exposing internal reasoning or raw markup |
| TTS and player | Synthesize and schedule speech | Playback cursor, not generated token count, defines what was heard |
| Event bus | Emit stage events and optional JSONL trace | Logs may contain sensitive transcripts and tool arguments |

## 1.3 Turn lifecycle

1. The input loop accumulates frames and uses VAD to identify speech.
2. Partial ASR runs periodically while speech is in progress.
3. When a turn detector considers the utterance likely complete, the system may
   start speculative LLM generation from the current partial transcript.
4. Final ASR runs after the turn ends. The partial and final transcripts are compared.
5. A matching speculative stream may be reused; a mismatch is cancelled and a new
   response is generated from the final transcript.
6. LLM text is stripped of supported reasoning tags, chunked, and queued for TTS.
7. Tool calls are assembled from the LLM stream and executed through the registry.
   The model is not an authority: tool names, arguments, and results are data that
   require validation and bounded execution.
8. Audio segments are played. If the caller interrupts, playback is faded/stopped
   and response work is cancelled.
9. Conversation history is committed using the player's playback cursor so that
   generated-but-unheard words are not treated as spoken.

## 1.4 Speculation and side effects

Speculation is a latency optimization, not an authorization mechanism. A partial
transcript can be wrong because ASR hypotheses evolve as more audio arrives. JOSIE's
intended invariant is:

> Speculative generation may prepare text, but tools that can change state run only
> after the final transcript has been accepted.

This invariant needs regression tests whenever orchestration is changed. Tests should
include a partial transcript that looks like “send it” and a final transcript that
resolves to “don't send it,” plus interruption while a tool is running. For tools
with irreversible external effects, cancellation alone is insufficient: use
idempotency keys, explicit confirmation, or a compensating transaction.

## 1.5 Tool orchestration

Independent tool calls may run concurrently to reduce wall-clock latency. Parallelism
is appropriate only when operations are independent and concurrency limits are
defined. The current built-ins include time lookup, web search, workspace file
operations, Python execution, and selected macOS actions. Their risk levels differ
substantially:

- Read-only lookups are lower risk but can still disclose sensitive context.
- Workspace writes are state-changing and must remain confined to a configured
  workspace.
- Running arbitrary Python is **not a sandbox** and inherits the process user's
  privileges.
- Calendar, application-launch, and volume tools act on the host and should be
  disabled or explicitly allow-listed in any remotely reachable deployment.

A production registry should define per-tool schemas, authorization policy,
idempotency, deadline, concurrency class, audit fields, and cancellation semantics.

## 1.6 State and history

There are at least three different text states: model-generated text, audio queued
for playback, and text estimated to have played. They must not be conflated. The
player's word boundaries are currently estimated from character proportions rather
than aligned to actual audio timestamps. Therefore “played-only history” is a useful
approximation, not phoneme- or word-level ground truth.

A stronger implementation should track sample-accurate playback offsets and use
forced alignment or TTS-provided timing when available. It should also define how
to represent a word cut in half, a fade-out tail, and audio sent to the device but
not physically heard by the listener.

## 1.7 Latency model

Measure latency as a set of stage timings rather than one aggregate number:

- **End-of-turn to first audio (E2FA):** final turn boundary to first audible assistant
  sample. This is the primary perceived-responsiveness measure.
- **Interruption-to-silence:** caller interruption detection to cessation of assistant
  output.
- **Tool latency:** tool dispatch to result, including timeout and cancellation.
- **ASR finalization latency:** end of captured speech to final transcript.
- **TTS real-time factor (RTF):** generated audio duration divided by synthesis wall time;
  values above 1 indicate synthesis faster than playback time.

Report p50, p95, and p99 by hardware, model/backend, utterance length, and cold/warm
state. Do not infer a latency SLA from a demo or a single average.
