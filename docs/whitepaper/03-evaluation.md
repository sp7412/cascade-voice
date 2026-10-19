# 3. Evaluation and production-readiness plan

## 3.1 Evaluation principle

Evaluate the whole workflow, not just ASR word error rate or LLM answer quality.
For a voice agent, success includes correct understanding, correct tool behavior,
timely speech, interruption handling, recovery from failure, and the final outcome.

A successful interaction is one where the user intent was correctly resolved and,
when a business action was requested, the resulting state was independently verified.
A plausible spoken confirmation is not proof that an action succeeded.

## 3.2 Test matrix

| Area | Test | Evidence / pass condition |
|---|---|---|
| Unit | Sentence chunking, think-tag filtering, turn decisions | Deterministic tests pass across chunk boundaries and edge cases |
| Unit | History trimming and playback cursor | No assistant turn is retained without its corresponding user turn; interrupted text is bounded to playback |
| Unit | Tool schema and unknown tool | Invalid calls are rejected before invoking a handler |
| Unit | Deadline and cancellation | Tool timeout returns a bounded error; cancellation leaves no task running |
| API | Missing/invalid bearer token | Protected route returns 401 |
| API | Upload larger than configured cap | Request is rejected before full body is buffered/read into application memory |
| API | Empty/invalid audio | Stable 4xx response; no traceback or internal path leak |
| API | Unsupported speech format | Stable 4xx response |
| Integration | ASR partial differs from final | Speculation is discarded and no tool executes from the partial |
| Integration | Barge-in during TTS | Output stops within a measured budget; stale response does not resume |
| Integration | Barge-in during tool call | Cancelled task is awaited; external side effects are reconciled |
| Integration | Model/network timeout | User receives a concise failure message; next turn still works |
| Hardware | Mic/speaker echo and double-talk | Measured false barge-in and missed barge-in rates under defined conditions |
| Load | Concurrent API calls | Memory, queue depth, and p95/p99 latency remain within declared budgets |
| Security | Workspace escape and hostile tool output | No out-of-scope filesystem access or instruction-following from tool output |

For each test, preserve the environment, model revisions, OS, hardware, package lock,
test seed, and results. A test that depends on a real model or audio device should be
marked separately from a deterministic unit test.

## 3.3 Latency and quality metrics

Report at least:

- End-of-turn-to-first-audio p50/p95/p99.
- Interruption-to-silence p50/p95/p99.
- ASR finalization latency and word error rate on a documented test set.
- Tool success, timeout, cancellation, and malformed-argument rates by tool.
- Task completion rate judged against a verified final state.
- Unintended action rate, especially for ambiguous or corrected utterances.
- TTS real-time factor and first-sentence synthesis latency.
- Resource use: peak RSS, CPU/GPU/ANE utilization, and queue depth.
- Cost per successful outcome for any hosted inference path.

Report both aggregate values and slices by noise, accent, utterance duration, network
condition, model/backend, and cold versus warm start. Never optimize average latency
at the cost of safety-critical tail failures.

## 3.4 Staged rollout

1. **Local deterministic tests:** no model downloads, no real side effects.
2. **Mocked integration:** fake ASR/LLM/TTS and fake tools; exercise cancellation races.
3. **Offline replay:** documented audio fixtures, expected transcripts, and interruption
   points; retain privacy-safe traces.
4. **Hardware-in-the-loop:** target microphone, speaker, CPU/GPU/ANE, and model versions.
5. **Shadow mode:** observe intents and proposed tool calls without executing actions.
6. **Scoped canary:** enable low-risk actions for a limited user cohort with a kill switch.
7. **Promotion:** require thresholds and rollback criteria to be met before broadening access.

Every rollout should be reversible. Keep configuration, model versions, prompts, and tool
policy versioned so an observed regression can be traced to a specific change.

## 3.5 Example promotion gates

These are proposed initial gates, not measured claims or universal standards. Tune them
to the product's risk profile and validate the measurement method first.

| Gate | Example threshold | Why it matters |
|---|---:|---|
| Critical unintended side effects | 0 in the defined test suite | A single incorrect action can outweigh a fluent answer |
| Tool argument schema compliance | 100% rejected or safely handled when invalid | Model output is untrusted input |
| Cancellation cleanup | No stale task or output after turn closure | Prevents old work contaminating new turns |
| API authorization | All negative auth tests pass | Prevents anonymous access when a key is configured |
| First audio | Declare a target after target-hardware baseline | Perceived responsiveness depends on the full pipeline |
| Interruption-to-silence | Declare a p95 target after measurement | User must be able to reclaim the floor |
| Task success | Independently verified outcome | Speech alone does not prove completion |
| Rollback | Tested within a declared time budget | Failed canaries must be reversible |

## 3.6 Production-readiness checklist

- [ ] Pin dependency and model revisions; document licenses and model terms.
- [ ] Define threat model, data flow, retention, and user consent for audio/logging.
- [ ] Configure API key, TLS, rate limits, concurrency limits, and request limits.
- [ ] Disable arbitrary code execution unless a real sandbox is configured.
- [ ] Validate all tool schemas and define authorization/idempotency per operation.
- [ ] Test timeout, cancellation, retry, duplicate request, and partial failure paths.
- [ ] Add metrics/tracing with redaction and operational dashboards.
- [ ] Benchmark target hardware with representative speech and acoustic conditions.
- [ ] Run shadow mode and scoped canary with explicit rollback triggers.
- [ ] Select a license before publishing or redistributing the repository.

## 3.7 Decision rule

Promote only when evidence supports the declared operating envelope. If a capability
has not been tested on the target environment, document it as unknown—not as passed.
