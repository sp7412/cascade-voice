# Operations runbook

This runbook describes safe operation of the current prototype. It is not a security
certification or a substitute for deployment-specific threat modeling.

## Local development

1. Use a dedicated virtual environment and a non-privileged OS account.
2. Start the server on loopback: `python -m josie serve --host 127.0.0.1`.
3. Keep credentials in environment variables and out of shell history where possible.
4. Leave arbitrary Python execution disabled unless you intentionally accept the risk.
5. Review JSONL traces before sharing them; they may contain transcripts and tool data.

## Non-loopback deployment

- Set a high-entropy `JOSIE_API_KEY` through the deployment secret manager.
- Use TLS at a reverse proxy or service mesh; bearer tokens over plaintext HTTP can be intercepted.
- Enforce request size, connection/read timeout, per-client rate, and concurrency limits at the edge.
- Run as a dedicated unprivileged account with a narrow filesystem view.
- Do not enable `JOSIE_ENABLE_PYTHON_TOOL` in a remotely reachable service. It is arbitrary
  code execution under the service account, not a sandbox.
- Restrict outbound network access and disable host-specific tools that are not required.
- Set log retention and permissions; never log authorization headers or audio payloads.
- Add a health check and graceful shutdown policy appropriate to your process manager.

## Incident response

1. Disable network ingress or stop the service.
2. Rotate `JOSIE_API_KEY` if exposure is suspected.
3. Preserve redacted operational logs and note deployed commit/model revisions.
4. Inspect tool audit events for unexpected state changes.
5. Reconcile external side effects manually; cancellation does not undo completed actions.
6. Re-enable only after the triggering condition has a regression test and a rollback plan.

## Release checklist

- [ ] CI and integration tests pass on the release commit.
- [ ] Dependency/model revisions and licenses are recorded.
- [ ] API auth, payload limits, and rate limits have been exercised.
- [ ] Cancellation and child-process cleanup have been tested.
- [ ] Hardware tests cover actual microphone, speakers, echo, and interruption.
- [ ] Metrics and rollback have been verified in the target environment.
