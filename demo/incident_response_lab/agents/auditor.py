"""Agent: auditor - compiles incident report and audit summary."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import ensure_sdk_on_path, wait_for_message

ensure_sdk_on_path()
from clove_sdk import CloveClient  # noqa: E402

AGENT_NAME = "auditor"


def main() -> int:
    client = CloveClient()
    if not client.connect():
        print("[auditor] ERROR: Failed to connect to Clove kernel")
        return 1

    incidents: list[str] = []
    triage_events: dict[str, dict] = {}
    remediation_events: dict[str, dict] = {}

    try:
        client.register_name(AGENT_NAME)
        init = wait_for_message(client, expected_type="init")
        run_id = init.get("run_id", "run_000")
        logs_dir = Path(init.get("logs_dir", "logs"))

        while True:
            message = wait_for_message(client)
            msg_type = message.get("type")

            if msg_type == "scan_complete":
                incidents = list(message.get("incidents", []))

            elif msg_type == "triage_event":
                incident_id = message.get("incident_id")
                if incident_id:
                    triage_events[incident_id] = message

            elif msg_type == "remediation_event":
                incident_id = message.get("incident_id")
                if incident_id:
                    remediation_events[incident_id] = message

            elif msg_type == "finalize":
                report = {
                    "run_id": run_id,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "incident_count": len(incidents),
                    "incidents": [],
                    "triage": triage_events,
                    "remediation": remediation_events,
                }

                for incident_id in incidents:
                    fetched = client.fetch(f"incident:{run_id}:{incident_id}")
                    if fetched.get("success") and fetched.get("exists"):
                        report["incidents"].append(fetched.get("value"))

                audit_log = client.get_audit_log(limit=50)
                if audit_log.get("success"):
                    report["audit_log_sample"] = audit_log.get("entries", [])[:10]

                report_path = logs_dir / run_id / "incident_report.json"
                payload = json.dumps(report, indent=2, sort_keys=True)
                client.write_file(str(report_path), payload)

                client.send_message({
                    "type": "audit_report",
                    "run_id": run_id,
                    "path": str(report_path)
                }, to_name="orchestrator")

            elif msg_type == "shutdown":
                break

    finally:
        client.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
