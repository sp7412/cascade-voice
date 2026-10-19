# JOSIE Voice Whitepaper

**Status:** engineering design and review document  
**Scope:** the current Cascade Voice / JOSIE prototype in this repository  
**Audience:** engineers evaluating, extending, testing, or operating the system

## Executive summary

JOSIE is a cascaded voice-agent prototype that composes distinct speech, language,
tool-execution, and playback stages. Its key design goal is to keep conversation
responsive without allowing speculative recognition to trigger side effects or
allowing unplayed generated text to silently become conversation history.

This is not a production-readiness certification. Model identifiers and optional
backend APIs may change; the microphone path, acoustic echo cancellation, barge-in,
speaker verification, deployment security, and end-to-end latency must be validated
on the target machine.

## Chapters

1. [Architecture and turn lifecycle](01-architecture.md) — component boundaries,
   streaming flow, speculation, tool orchestration, and playback-aware history.
2. [Reliability, security, and failure handling](02-reliability-security.md) —
   threat model, cancellation semantics, HTTP safeguards, operational risks, and
   concrete acceptance criteria.
3. [Evaluation and production-readiness plan](03-evaluation.md) — test matrix,
   latency definitions, business outcomes, staged rollout, and promotion gates.
4. [Operations runbook](04-operations-runbook.md) — safe deployment, incident response,
   operational checks, and release checklist.

## Five takeaways

- The system is a **cascade**, not a single end-to-end speech model.
- Partial ASR may improve perceived latency; it must not authorize side effects.
- Playback state is part of conversation state: generated text is not necessarily
  text the caller heard.
- Tool calls are an execution boundary and require validation, deadlines,
  cancellation, least privilege, and auditable results.
- Production claims require measured evidence from the complete path, not just
  model benchmarks or passing unit tests.

## Reading the status correctly

“Implemented” means code exists in this repository. “Unit-tested” means a focused
automated test exercises the behavior. Neither implies hardware validation,
security assurance, or production suitability. The evaluation chapter defines
what additional evidence is needed.
