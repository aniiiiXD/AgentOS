"""Agent: anomaly_triager - assigns severity and escalation decisions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import ensure_sdk_on_path, wait_for_message

ensure_sdk_on_path()
from clove_sdk import CloveClient  # noqa: E402

AGENT_NAME = "anomaly_triager"

SEVERITY_PRIORITY = {
    "low": "P4",
    "medium": "P3",
    "high": "P2",
    "critical": "P1",
}


def main() -> int:
    client = CloveClient()
    if not client.connect():
        print("[anomaly_triager] ERROR: Failed to connect to Clove kernel")
        return 1

    try:
        client.register_name(AGENT_NAME)
        init = wait_for_message(client, expected_type="init")
        run_id = init.get("run_id", "run_000")
        playbook = init.get("remediation_playbook", {})

        while True:
            message = wait_for_message(client)
            msg_type = message.get("type")

            if msg_type == "anomaly_detected":
                incident = message.get("incident", {})
                incident_id = incident.get("id")
                severity = incident.get("severity", "low")
                priority = SEVERITY_PRIORITY.get(severity, "P4")

                triage = {
                    "incident_id": incident_id,
                    "run_id": run_id,
                    "severity": severity,
                    "priority": priority,
                    "status": "triaged",
                }
                client.store(f"triage:{run_id}:{incident_id}", triage, scope="global")

                client.send_message({
                    "type": "triage_event",
                    "incident_id": incident_id,
                    "severity": severity,
                    "priority": priority,
                    "status": "triaged"
                }, to_name="auditor")

                if severity in {"high", "critical"}:
                    action = playbook.get(incident.get("type", ""))
                    client.send_message({
                        "type": "remediate",
                        "incident": incident,
                        "triage": triage,
                        "action": action,
                    }, to_name="remediation_executor")
                else:
                    client.send_message({
                        "type": "remediation_event",
                        "incident_id": incident_id,
                        "status": "not_required",
                        "action": None
                    }, to_name="auditor")

            elif msg_type == "shutdown":
                break

    finally:
        client.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
