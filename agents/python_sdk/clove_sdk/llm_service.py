#!/usr/bin/env python3
"""
Local LLM service wrapper for the SDK.

Runs agents/llm_service/llm_service.py as a long-lived subprocess and returns JSON output.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
import threading
import atexit
from pathlib import Path
from typing import Any, Dict, Optional


def _find_llm_service() -> Optional[Path]:
    override = os.environ.get("CLOVE_LLM_SERVICE_PATH")
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return path

    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "llm_service" / "llm_service.py",  # agents/llm_service/llm_service.py
        here.parents[3] / "agents" / "llm_service" / "llm_service.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


class _LLMServiceProcess:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()

    def _start(self) -> Optional[str]:
        script_path = _find_llm_service()
        if not script_path:
            return "LLM service not found. Set CLOVE_LLM_SERVICE_PATH or install agents/llm_service."

        self._proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return None

    def _is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if not self._is_running():
                err = self._start()
                if err:
                    return {"success": False, "error": err, "content": ""}

            assert self._proc is not None
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None

            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except Exception as exc:
                return {"success": False, "error": str(exc), "content": ""}

            line = self._proc.stdout.readline()
            if not line:
                err = ""
                if self._proc.stderr:
                    err = self._proc.stderr.read().strip()
                return {"success": False, "error": err or "No response from LLM service", "content": ""}

            try:
                return json.loads(line.strip())
            except json.JSONDecodeError:
                return {"success": False, "error": "Invalid JSON from LLM service", "content": line.strip()}

    def shutdown(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
        self._proc = None


_CLIENT = _LLMServiceProcess()
atexit.register(_CLIENT.shutdown)


def call_llm_service(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _CLIENT.call(payload)
