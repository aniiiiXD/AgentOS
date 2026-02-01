# Clove Incident Response Lab (Agentic Demo)

This demo models a multi-agent incident response workflow on Clove. It focuses on agentic coordination, IPC, permission boundaries, and auditability — not ML training.

## Agents

- **log_watcher**: reads a system log via kernel file syscalls, detects anomalies.
- **anomaly_triager**: assigns severity/priority and decides whether to remediate.
- **remediation_executor**: runs a whitelisted command through the kernel exec syscall.
- **auditor**: compiles an incident report + audit log sample.

## What it demonstrates

- **IPC**: agents send incident, triage, and remediation messages via kernel IPC.
- **Permissions**: remediation is the only agent allowed to exec, and only `python3` is whitelisted.
- **Quotas/limits**: per-agent CPU/memory limits are applied at spawn.
- **Auditability**: execution recording and audit log export are available from the orchestrator.

## Prereqs

- Clove kernel running (`./build/clove_kernel` or `sudo ./build/clove_kernel`)
- Python 3.10+

## Run

From repo root:

```bash
python3 demo/incident_response_lab/main.py --record-execution --dump-audit
```

## Outputs

Artifacts and logs are written to:

- `demo/incident_response_lab/artifacts/<run_id>/`
- `demo/incident_response_lab/logs/<run_id>/`

Key files:

- `system.log` (input log stream)
- `incident_report.json` (auditor summary)
- `execution_recording.json` (kernel exec recording, if enabled)
- `audit_log.json` (kernel audit log sample, if enabled)

## Customize the scenario

Edit `demo/incident_response_lab/configs/scenario.json` to add new log lines, detection rules, or remediation playbook entries.
