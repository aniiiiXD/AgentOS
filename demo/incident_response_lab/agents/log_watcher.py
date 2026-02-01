"""Agent: log_watcher - scans logs and emits anomaly alerts."""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import ensure_sdk_on_path, wait_for_message

ensure_sdk_on_path()
from clove_sdk import CloveClient  # noqa: E402

AGENT_NAME = "log_watcher"


def extract_value(line: str, key: str) -> str | None:
    match = re.search(rf"{re.escape(key)}=([^\s\"]+)", line)
    if match:
        return match.group(1).strip('"')
    return None


def detect_incident(line: str, rules: dict) -> tuple[str, dict] | None:
    for keyword, rule in rules.items():
        if keyword in line:
            rule_data = rule if isinstance(rule, dict) else {"severity": rule}
            return keyword, rule_data
    return None


def main() -> int:
    client = CloveClient()
    if not client.connect():
        print("[log_watcher] ERROR: Failed to connect to Clove kernel")
        return 1

    try:
        client.register_name(AGENT_NAME)
        init = wait_for_message(client, expected_type="init")
        run_id = init.get("run_id", "run_000")
        rules = init.get("rules", {})
        reply_to = init.get("reply_to", "orchestrator")

        while True:
            message = wait_for_message(client)
            msg_type = message.get("type")

            if msg_type == "scan_logs":
                log_path = message.get("log_path")
                if not log_path:
                    client.send_message({
                        "type": "scan_complete",
                        "run_id": run_id,
                        "count": 0,
                        "incidents": [],
                        "error": "missing log_path"
                    }, to_name=reply_to)
                    continue

                read_result = client.read_file(log_path)
                if not read_result.get("success"):
                    client.send_message({
                        "type": "scan_complete",
                        "run_id": run_id,
                        "count": 0,
                        "incidents": [],
                        "error": read_result.get("error", "read failed")
                    }, to_name=reply_to)
                    continue

                content = read_result.get("content", "")
                incidents = []
                counter = 0
                for line in content.splitlines():
                    detected = detect_incident(line, rules)
                    if not detected:
                        continue
                    counter += 1
                    incident_type, rule_data = detected
                    incident_id = f"inc_{run_id}_{counter:02d}"
                    incident = {
                        "id": incident_id,
                        "run_id": run_id,
                        "type": incident_type,
                        "severity": rule_data.get("severity", "low"),
                        "title": rule_data.get("title", incident_type),
                        "line": line,
                        "source_ip": extract_value(line, "src"),
                        "user": extract_value(line, "user"),
                        "detected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "detected"
                    }
                    incidents.append(incident_id)
                    client.store(f"incident:{run_id}:{incident_id}", incident, scope="global")
                    client.send_message({
                        "type": "anomaly_detected",
                        "incident": incident
                    }, to_name="anomaly_triager")

                client.send_message({
                    "type": "scan_complete",
                    "run_id": run_id,
                    "count": len(incidents),
                    "incidents": incidents
                }, to_name=reply_to)

            elif msg_type == "shutdown":
                break

    finally:
        client.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
