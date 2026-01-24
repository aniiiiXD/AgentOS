#!/usr/bin/env python3
"""
Clove Remote Client SDK

Client library for connecting to a remote Clove kernel through a relay server.
"""

import json
import base64
import struct
import asyncio
import threading
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import IntEnum
from concurrent.futures import Future
import queue

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ImportError:
    raise ImportError("websockets library required. Run: pip install clove-sdk[remote]")

from .client import SyscallOp, Message, MAGIC_BYTES, HEADER_SIZE


class RemoteAgentClient:
    """
    Client for connecting to a remote Clove kernel via relay server.
    API is compatible with CloveClient for easy migration.
    """

    def __init__(self, relay_url: str, agent_name: str, agent_token: str,
                 target_machine: str, reconnect: bool = True):
        self.relay_url = relay_url
        self.agent_name = agent_name
        self.agent_token = agent_token
        self.target_machine = target_machine
        self.reconnect = reconnect

        self._ws: Optional[WebSocketClientProtocol] = None
        self._agent_id: int = 0
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        self._pending_responses: Dict[int, Future] = {}
        self._response_queue = queue.Queue()
        self._request_id = 0

    def connect(self) -> bool:
        """Connect to the relay server and authenticate"""
        if self._connected:
            return True

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

        future = asyncio.run_coroutine_threadsafe(self._connect_async(), self._loop)
        try:
            return future.result(timeout=30)
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from the relay server"""
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._disconnect_async(), self._loop)
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
        self._connected = False

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_async(self) -> bool:
        try:
            self._ws = await websockets.connect(
                self.relay_url, ping_interval=30, ping_timeout=10
            )

            auth_msg = {
                "type": "agent_auth",
                "name": self.agent_name,
                "token": self.agent_token,
                "target_machine": self.target_machine
            }
            await self._ws.send(json.dumps(auth_msg))

            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            data = json.loads(response)

            if data.get("type") == "auth_ok":
                self._agent_id = data.get("agent_id", 0)
                self._connected = True
                asyncio.create_task(self._message_loop())
                return True
            else:
                raise Exception(data.get("error", "Authentication failed"))

        except Exception as e:
            self._connected = False
            if self._ws:
                await self._ws.close()
                self._ws = None
            raise

    async def _disconnect_async(self):
        self._connected = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def _message_loop(self):
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        base_delay = 1.0

        while self._connected or (self.reconnect and reconnect_attempts < max_reconnect_attempts):
            try:
                if self._ws is None:
                    break

                async for message in self._ws:
                    reconnect_attempts = 0
                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON from relay: {e}")
                    except Exception as e:
                        print(f"Error handling message: {e}")

            except websockets.ConnectionClosed as e:
                self._connected = False
                if self.reconnect and reconnect_attempts < max_reconnect_attempts:
                    reconnect_attempts += 1
                    delay = base_delay * (2 ** (reconnect_attempts - 1))
                    print(f"Connection closed (code={e.code}), reconnecting in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    try:
                        if await self._connect_async():
                            continue
                    except Exception as re:
                        print(f"Reconnection failed: {re}")
                else:
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                break

        self._connected = False

    async def _handle_message(self, data: dict):
        msg_type = data.get("type")

        if msg_type == "response":
            opcode = data.get("opcode", 0)
            payload_b64 = data.get("payload", "")
            payload = base64.b64decode(payload_b64) if payload_b64 else b""

            self._response_queue.put(Message(
                agent_id=self._agent_id,
                opcode=SyscallOp(opcode),
                payload=payload
            ))
        elif msg_type == "kernel_disconnected":
            self._connected = False
            print(f"Kernel disconnected: {data.get('machine_id')}")
        elif msg_type == "error":
            print(f"Relay error: {data.get('error')}")

    def call(self, opcode: SyscallOp, payload: bytes | str = b'') -> Optional[Message]:
        """Send a syscall and wait for response"""
        if not self._connected or not self._loop:
            return None

        if isinstance(payload, str):
            payload = payload.encode('utf-8')

        while not self._response_queue.empty():
            try:
                self._response_queue.get_nowait()
            except queue.Empty:
                break

        future = asyncio.run_coroutine_threadsafe(
            self._send_syscall(opcode, payload), self._loop
        )

        try:
            future.result(timeout=5)
        except Exception as e:
            print(f"Failed to send syscall: {e}")
            return None

        try:
            response = self._response_queue.get(timeout=60)
            return response
        except queue.Empty:
            print("Timeout waiting for response")
            return None

    async def _send_syscall(self, opcode: SyscallOp, payload: bytes):
        if not self._ws:
            return

        msg = {
            "type": "syscall",
            "opcode": int(opcode),
            "payload": base64.b64encode(payload).decode() if payload else ""
        }
        await self._ws.send(json.dumps(msg))

    # High-level API (compatible with CloveClient)

    def echo(self, message: str) -> Optional[str]:
        response = self.call(SyscallOp.SYS_NOOP, message)
        return response.payload_str if response else None

    def think(self, prompt: str, image: bytes = None,
              image_mime_type: str = "image/jpeg",
              system_instruction: str = None,
              thinking_level: str = None,
              temperature: float = None,
              model: str = None) -> dict:
        payload = {"prompt": prompt}
        if image:
            payload["image"] = {"data": base64.b64encode(image).decode(), "mime_type": image_mime_type}
        if system_instruction:
            payload["system_instruction"] = system_instruction
        if thinking_level:
            payload["thinking_level"] = thinking_level
        if temperature is not None:
            payload["temperature"] = temperature
        if model:
            payload["model"] = model

        response = self.call(SyscallOp.SYS_THINK, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": True, "content": response.payload_str}
        return {"success": False, "content": "", "error": "No response"}

    def exec(self, command: str, cwd: str = None, timeout: int = 30) -> dict:
        payload = {"command": command, "timeout": timeout}
        if cwd:
            payload["cwd"] = cwd
        response = self.call(SyscallOp.SYS_EXEC, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response"}

    def read_file(self, path: str) -> dict:
        response = self.call(SyscallOp.SYS_READ, json.dumps({"path": path}))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response"}

    def write_file(self, path: str, content: str, mode: str = "write") -> dict:
        payload = {"path": path, "content": content, "mode": mode}
        response = self.call(SyscallOp.SYS_WRITE, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response"}

    def spawn(self, name: str, script: str, sandboxed: bool = True,
              network: bool = False, limits: dict = None) -> Optional[dict]:
        payload = {"name": name, "script": script, "sandboxed": sandboxed, "network": network}
        if limits:
            payload["limits"] = limits
        response = self.call(SyscallOp.SYS_SPAWN, json.dumps(payload))
        if response:
            return json.loads(response.payload_str)
        return None

    def kill(self, name: str = None, agent_id: int = None) -> bool:
        payload = {}
        if name:
            payload["name"] = name
        elif agent_id:
            payload["id"] = agent_id
        else:
            return False
        response = self.call(SyscallOp.SYS_KILL, json.dumps(payload))
        if response:
            result = json.loads(response.payload_str)
            return result.get("killed", False)
        return False

    def list_agents(self) -> list:
        response = self.call(SyscallOp.SYS_LIST)
        if response:
            return json.loads(response.payload_str)
        return []

    def store(self, key: str, value, scope: str = "global", ttl: int = None) -> dict:
        payload = {"key": key, "value": value, "scope": scope}
        if ttl is not None:
            payload["ttl"] = ttl
        response = self.call(SyscallOp.SYS_STORE, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response"}

    def fetch(self, key: str) -> dict:
        response = self.call(SyscallOp.SYS_FETCH, json.dumps({"key": key}))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response"}

    def http(self, url: str, method: str = "GET", headers: dict = None,
             body: str = None, timeout: int = 30) -> dict:
        payload = {"url": url, "method": method, "timeout": timeout}
        if headers:
            payload["headers"] = headers
        if body:
            payload["body"] = body
        response = self.call(SyscallOp.SYS_HTTP, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response"}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def connect_remote(relay_url: str, agent_name: str, agent_token: str,
                  target_machine: str) -> RemoteAgentClient:
    """Create and connect a remote client"""
    client = RemoteAgentClient(
        relay_url=relay_url, agent_name=agent_name,
        agent_token=agent_token, target_machine=target_machine
    )
    if not client.connect():
        raise ConnectionError(f"Failed to connect to relay at {relay_url}")
    return client
