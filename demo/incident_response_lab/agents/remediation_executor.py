"""Agent: remediation_executor - performs safe, whitelisted remediation actions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import ensure_sdk_on_path, safe_py_string, wait_for_message

ensure_sdk_on_path()
from clove_sdk import CloveClient  # noqa: E402

AGENT_NAME = "remediation_executor"


def build_command(action_path: Path, message: str) -> str:
    safe_message = safe_py_string(message)
    safe_path = safe_py_string(str(action_path))
    return (
        "python3 -c \""
        "from pathlib import Path; "
        f"p=Path('{safe_path}'); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        "with p.open('a', encoding='utf-8') as f: "
        f"f.write('{safe_message}\\n')\""
    )


def main() -> int:
    client = CloveClient()
    if not client.connect():
        print("[remediation_executor] ERROR: Failed to connect to Clove kernel")
        return 1

    try:
        client.register_name(AGENT_NAME)
        init = wait_for_message(client, expected_type="init")
        run_id = init.get("run_id", "run_000")
        artifacts_dir = Path(init.get("artifacts_dir", "artifacts"))

        while True:
            message = wait_for_message(client)
            msg_type = message.get("type")

            if msg_type == "remediate":
                incident = message.get("incident", {})
                action = message.get("action")
                incident_id = incident.get("id", "unknown")

                if not action:
                    client.send_message({
                        "type": "remediation_event",
                        "incident_id": incident_id,
                        "status": "no_playbook",
                        "action": None
                    }, to_name="auditor")
                    continue

                action_name = action.get("action", "unknown")
                action_desc = action.get("description", "")
                action_path = artifacts_dir / run_id / "remediation_actions.log"
                summary = f"{incident_id} {action_name} {action_desc}".strip()
                command = build_command(action_path, summary)

                result = client.exec(command, timeout=5)
                status = "ok" if result.get("success") else "failed"

                record = {
                    "incident_id": incident_id,
                    "run_id": run_id,
                    "action": action_name,
                    "description": action_desc,
                    "command": command,
                    "exec_result": result,
                    "status": status,
                }
                client.store(f"remediation:{run_id}:{incident_id}", record, scope="global")

                client.send_message({
                    "type": "remediation_event",
                    "incident_id": incident_id,
                    "status": status,
                    "action": action_name,
                    "exec_result": {
                        "success": result.get("success"),
                        "exit_code": result.get("exit_code")
                    }
                }, to_name="auditor")

            elif msg_type == "shutdown":
                break

    finally:
        client.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
