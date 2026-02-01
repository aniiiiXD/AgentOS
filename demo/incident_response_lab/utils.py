"""Shared helpers for the incident response lab demo."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


def lab_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return lab_root().parents[1]


def ensure_sdk_on_path() -> None:
    sdk_path = repo_root() / "agents" / "python_sdk"
    if sdk_path.exists() and str(sdk_path) not in sys.path:
        sys.path.insert(0, str(sdk_path))


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    suffix = path.suffix.lower()
    raw = path.read_text()
    if suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {"value": data}
    return {}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def log_line(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def normalize_limits(limits: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(limits) if limits else {}
    if "memory_mb" in normalized and "memory" not in normalized:
        normalized["memory"] = int(normalized["memory_mb"]) * 1024 * 1024
    if "cpu" in normalized and "cpu_quota" not in normalized:
        normalized["cpu_quota"] = int(normalized["cpu"]) * 100000
    return normalized


def wait_for_message(
    client,
    poll_interval: float = 0.2,
    expected_type: str | None = None,
) -> Dict[str, Any]:
    while True:
        result = client.recv_messages()
        for msg in result.get("messages", []):
            payload = msg.get("message", {})
            if not payload:
                continue
            if expected_type and payload.get("type") != expected_type:
                continue
            return payload
        time.sleep(poll_interval)


def safe_py_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")
