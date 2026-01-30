#!/usr/bin/env python3
"""
Clove Python SDK

Client library for communicating with the Clove kernel via Unix domain sockets.
"""

import socket
import struct
from enum import IntEnum
from typing import Optional, Tuple
from dataclasses import dataclass


# Protocol constants
MAGIC_BYTES = 0x41474E54  # "AGNT" in hex
HEADER_SIZE = 17
MAX_PAYLOAD_SIZE = 1024 * 1024  # 1MB


class SyscallOp(IntEnum):
    """System call operations"""
    SYS_NOOP = 0x00   # For testing / echo
    SYS_THINK = 0x01  # Send prompt to LLM
    SYS_EXEC = 0x02   # Execute shell command
    SYS_READ = 0x03   # Read file
    SYS_WRITE = 0x04  # Write file
    SYS_SPAWN = 0x10  # Spawn a sandboxed agent
    SYS_KILL = 0x11   # Kill an agent
    SYS_LIST = 0x12   # List running agents
    SYS_PAUSE = 0x14  # Pause an agent
    SYS_RESUME = 0x15 # Resume a paused agent
    # IPC - Inter-Agent Communication
    SYS_SEND = 0x20       # Send message to another agent
    SYS_RECV = 0x21       # Receive pending messages
    SYS_BROADCAST = 0x22  # Broadcast message to all agents
    SYS_REGISTER = 0x23   # Register agent name
    # State Store
    SYS_STORE = 0x30      # Store key-value pair
    SYS_FETCH = 0x31      # Retrieve value by key
    SYS_DELETE = 0x32     # Delete a key
    SYS_KEYS = 0x33       # List keys with optional prefix
    # Permissions
    SYS_GET_PERMS = 0x40  # Get own permissions
    SYS_SET_PERMS = 0x41  # Set agent permissions
    # Network
    SYS_HTTP = 0x50       # Make HTTP request
    # Events (Pub/Sub)
    SYS_SUBSCRIBE = 0x60    # Subscribe to event types
    SYS_UNSUBSCRIBE = 0x61  # Unsubscribe from events
    SYS_POLL_EVENTS = 0x62  # Get pending events
    SYS_EMIT = 0x63         # Emit custom event
    # World Simulation
    SYS_WORLD_CREATE = 0xA0    # Create world from config
    SYS_WORLD_DESTROY = 0xA1   # Destroy world
    SYS_WORLD_LIST = 0xA2      # List active worlds
    SYS_WORLD_JOIN = 0xA3      # Join agent to world
    SYS_WORLD_LEAVE = 0xA4     # Remove agent from world
    SYS_WORLD_EVENT = 0xA5     # Inject chaos event
    SYS_WORLD_STATE = 0xA6     # Get world metrics
    SYS_WORLD_SNAPSHOT = 0xA7  # Save world state
    SYS_WORLD_RESTORE = 0xA8   # Restore from snapshot
    # Remote Connectivity (Tunnel)
    SYS_TUNNEL_CONNECT = 0xB0      # Connect kernel to relay server
    SYS_TUNNEL_DISCONNECT = 0xB1   # Disconnect from relay
    SYS_TUNNEL_STATUS = 0xB2       # Get tunnel connection status
    SYS_TUNNEL_LIST_REMOTES = 0xB3 # List connected remote agents
    SYS_TUNNEL_CONFIG = 0xB4       # Configure tunnel settings
    # Metrics
    SYS_METRICS_SYSTEM = 0xC0      # Get system-wide metrics
    SYS_METRICS_AGENT = 0xC1       # Get metrics for specific agent
    SYS_METRICS_ALL_AGENTS = 0xC2  # Get metrics for all agents
    SYS_METRICS_CGROUP = 0xC3      # Get cgroup metrics
    # Audit Logging
    SYS_GET_AUDIT_LOG = 0x76       # Get audit log entries
    SYS_SET_AUDIT_CONFIG = 0x77    # Configure audit logging
    # Execution Recording & Replay
    SYS_RECORD_START = 0x70        # Start recording execution
    SYS_RECORD_STOP = 0x71         # Stop recording
    SYS_RECORD_STATUS = 0x72       # Get recording status
    SYS_REPLAY_START = 0x73        # Start replay
    SYS_REPLAY_STATUS = 0x74       # Get replay status
    # Async Results
    SYS_ASYNC_POLL = 0x80          # Poll async syscall results
    # Kernel info / capabilities
    SYS_LLM_REPORT = 0xF0          # Report SDK LLM usage to kernel
    SYS_HELLO = 0xFE               # Handshake / capability query
    SYS_EXIT = 0xFF   # Graceful shutdown


@dataclass
class Message:
    """Clove message"""
    agent_id: int
    opcode: SyscallOp
    payload: bytes

    def serialize(self) -> bytes:
        """Serialize message to wire format"""
        header = struct.pack(
            '<IIBQ',  # little-endian: uint32, uint32, uint8, uint64
            MAGIC_BYTES,
            self.agent_id,
            self.opcode,
            len(self.payload)
        )
        return header + self.payload

    @classmethod
    def deserialize(cls, data: bytes) -> Optional['Message']:
        """Deserialize message from wire format"""
        if len(data) < HEADER_SIZE:
            return None

        magic, agent_id, opcode, payload_size = struct.unpack('<IIBQ', data[:HEADER_SIZE])

        if magic != MAGIC_BYTES:
            return None

        if payload_size > MAX_PAYLOAD_SIZE:
            return None

        if len(data) < HEADER_SIZE + payload_size:
            return None

        payload = data[HEADER_SIZE:HEADER_SIZE + payload_size]
        return cls(agent_id=agent_id, opcode=SyscallOp(opcode), payload=payload)

    @property
    def payload_str(self) -> str:
        """Get payload as string"""
        return self.payload.decode('utf-8', errors='replace')


class CloveClient:
    """Client for communicating with Clove kernel"""

    def __init__(self, socket_path: str = '/tmp/clove.sock'):
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None
        self._agent_id = 0

    @property
    def agent_id(self) -> int:
        """Get the agent ID assigned by the kernel"""
        return self._agent_id

    def connect(self) -> bool:
        """Connect to the Clove kernel"""
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.connect(self.socket_path)
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from the kernel"""
        if self._sock:
            self._sock.close()
            self._sock = None

    def send(self, opcode: SyscallOp, payload: bytes | str = b'') -> bool:
        """Send a message to the kernel"""
        if not self._sock:
            return False

        if isinstance(payload, str):
            payload = payload.encode('utf-8')

        msg = Message(agent_id=self._agent_id, opcode=opcode, payload=payload)
        try:
            self._sock.sendall(msg.serialize())
            return True
        except Exception as e:
            print(f"Send failed: {e}")
            return False

    def recv(self) -> Optional[Message]:
        """Receive a message from the kernel"""
        if not self._sock:
            return None

        try:
            # Read header first
            header_data = self._recv_exact(HEADER_SIZE)
            if not header_data:
                return None

            # Parse header to get payload size
            magic, agent_id, opcode, payload_size = struct.unpack('<IIBQ', header_data)

            if magic != MAGIC_BYTES:
                print(f"Invalid magic bytes: 0x{magic:08x}")
                return None

            # Read payload
            payload = b''
            if payload_size > 0:
                payload = self._recv_exact(payload_size)
                if not payload:
                    return None

            # Update our agent ID from response
            self._agent_id = agent_id

            return Message(agent_id=agent_id, opcode=SyscallOp(opcode), payload=payload)
        except Exception as e:
            print(f"Receive failed: {e}")
            return None

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def call(self, opcode: SyscallOp, payload: bytes | str = b'') -> Optional[Message]:
        """Send a message and wait for response"""
        if not self.send(opcode, payload):
            return None
        return self.recv()

    # Convenience methods
    def hello(self) -> dict:
        """Query kernel version and capabilities."""
        import json
        response = self.call(SyscallOp.SYS_HELLO, "{}")
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def echo(self, message: str) -> Optional[str]:
        """Echo a message (for testing)"""
        response = self.call(SyscallOp.SYS_NOOP, message)
        return response.payload_str if response else None

    def noop(self, message: str) -> Optional[str]:
        """Alias for echo - send a NOOP message (for testing)"""
        return self.echo(message)

    def think(self, prompt: str,
              image: bytes = None,
              image_mime_type: str = "image/jpeg",
              system_instruction: str = None,
              thinking_level: str = None,
              temperature: float = None,
              model: str = None,
              async_: bool = False,
              request_id: int = None) -> dict:
        """Send a prompt to the LLM via local LLM service (not the kernel)."""
        import base64
        import json
        from .llm_service import call_llm_service

        payload = {"prompt": prompt}

        if image:
            payload["image"] = {
                "data": base64.b64encode(image).decode(),
                "mime_type": image_mime_type
            }

        if system_instruction:
            payload["system_instruction"] = system_instruction

        if thinking_level:
            payload["thinking_level"] = thinking_level

        if temperature is not None:
            payload["temperature"] = temperature

        if model:
            payload["model"] = model
        if async_ or request_id is not None:
            payload["async"] = False

        result = call_llm_service(payload)

        # Report LLM usage to kernel if connected
        if self._sock and result.get("success"):
            tokens = int(result.get("tokens", 0) or 0)
            report = {"tokens": tokens, "success": True}
            self.call(SyscallOp.SYS_LLM_REPORT, json.dumps(report))

        return result

    def exit(self) -> bool:
        """Request graceful exit"""
        response = self.call(SyscallOp.SYS_EXIT)
        return response is not None

    def spawn(self, name: str, script: str, sandboxed: bool = True,
              network: bool = False, limits: dict = None,
              restart_policy: str = "never",
              max_restarts: int = 5,
              restart_window: int = 300) -> Optional[dict]:
        """Spawn a new sandboxed agent."""
        import json
        payload = {
            "name": name,
            "script": script,
            "sandboxed": sandboxed,
            "network": network,
            "restart_policy": restart_policy,
            "max_restarts": max_restarts,
            "restart_window": restart_window
        }
        if limits:
            payload["limits"] = limits

        response = self.call(SyscallOp.SYS_SPAWN, json.dumps(payload))
        if response:
            return json.loads(response.payload_str)
        return None

    def kill(self, name: str = None, agent_id: int = None) -> bool:
        """Kill a running agent"""
        import json
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

    def pause(self, name: str = None, agent_id: int = None) -> bool:
        """Pause a running agent (SIGSTOP)"""
        import json
        payload = {}
        if name:
            payload["name"] = name
        elif agent_id:
            payload["id"] = agent_id
        else:
            return False

        response = self.call(SyscallOp.SYS_PAUSE, json.dumps(payload))
        if response:
            result = json.loads(response.payload_str)
            return result.get("success", False)
        return False

    def resume(self, name: str = None, agent_id: int = None) -> bool:
        """Resume a paused agent (SIGCONT)"""
        import json
        payload = {}
        if name:
            payload["name"] = name
        elif agent_id:
            payload["id"] = agent_id
        else:
            return False

        response = self.call(SyscallOp.SYS_RESUME, json.dumps(payload))
        if response:
            result = json.loads(response.payload_str)
            return result.get("success", False)
        return False

    def list_agents(self) -> list:
        """List all running agents"""
        import json
        response = self.call(SyscallOp.SYS_LIST)
        if response:
            return json.loads(response.payload_str)
        return []

    def exec(self, command: str, cwd: str = None, timeout: int = 30,
             async_: bool = False, request_id: int = None) -> dict:
        """Execute a shell command."""
        import json
        payload = {
            "command": command,
            "timeout": timeout,
            "async": async_
        }
        if cwd:
            payload["cwd"] = cwd
        if request_id is not None:
            payload["request_id"] = request_id

        response = self.call(SyscallOp.SYS_EXEC, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "stdout": "", "stderr": response.payload_str, "exit_code": -1}
        return {"success": False, "stdout": "", "stderr": "No response from kernel", "exit_code": -1}

    def read_file(self, path: str) -> dict:
        """Read a file's contents."""
        import json
        payload = {"path": path}

        response = self.call(SyscallOp.SYS_READ, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "content": "", "size": 0, "error": response.payload_str}
        return {"success": False, "content": "", "size": 0, "error": "No response from kernel"}

    def write_file(self, path: str, content: str, mode: str = "write") -> dict:
        """Write content to a file."""
        import json
        payload = {
            "path": path,
            "content": content,
            "mode": mode
        }

        response = self.call(SyscallOp.SYS_WRITE, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "bytes_written": 0, "error": response.payload_str}
        return {"success": False, "bytes_written": 0, "error": "No response from kernel"}

    # IPC - Inter-Agent Communication

    def register_name(self, name: str) -> dict:
        """Register this agent with a name for IPC."""
        import json
        payload = {"name": name}

        response = self.call(SyscallOp.SYS_REGISTER, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def send_message(self, message: dict, to: int = None, to_name: str = None) -> dict:
        """Send a message to another agent."""
        import json
        payload = {"message": message}

        if to is not None:
            payload["to"] = to
        if to_name is not None:
            payload["to_name"] = to_name

        response = self.call(SyscallOp.SYS_SEND, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def recv_messages(self, max_messages: int = 10) -> dict:
        """Receive pending messages from other agents."""
        import json
        payload = {"max": max_messages}

        response = self.call(SyscallOp.SYS_RECV, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "messages": [], "count": 0, "error": response.payload_str}
        return {"success": False, "messages": [], "count": 0, "error": "No response from kernel"}

    def broadcast(self, message: dict, include_self: bool = False) -> dict:
        """Broadcast a message to all registered agents."""
        import json
        payload = {
            "message": message,
            "include_self": include_self
        }

        response = self.call(SyscallOp.SYS_BROADCAST, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "delivered_count": 0, "error": response.payload_str}
        return {"success": False, "delivered_count": 0, "error": "No response from kernel"}

    # Permissions

    def get_permissions(self) -> dict:
        """Get this agent's permissions."""
        import json
        response = self.call(SyscallOp.SYS_GET_PERMS, "{}")
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def set_permissions(self, permissions: dict = None, level: str = None,
                       agent_id: int = None) -> dict:
        """Set agent permissions."""
        import json
        payload = {}

        if permissions:
            payload["permissions"] = permissions
        if level:
            payload["level"] = level
        if agent_id is not None:
            payload["agent_id"] = agent_id

        response = self.call(SyscallOp.SYS_SET_PERMS, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    # State Store

    def store(self, key: str, value, scope: str = "global", ttl: int = None) -> dict:
        """Store a key-value pair in the shared state store."""
        import json
        payload = {
            "key": key,
            "value": value,
            "scope": scope
        }
        if ttl is not None:
            payload["ttl"] = ttl

        response = self.call(SyscallOp.SYS_STORE, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def fetch(self, key: str) -> dict:
        """Fetch a value from the shared state store."""
        import json
        payload = {"key": key}

        response = self.call(SyscallOp.SYS_FETCH, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def delete_key(self, key: str) -> dict:
        """Delete a key from the shared state store."""
        import json
        payload = {"key": key}

        response = self.call(SyscallOp.SYS_DELETE, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def list_keys(self, prefix: str = "") -> dict:
        """List keys in the shared state store."""
        import json
        payload = {"prefix": prefix} if prefix else {}

        response = self.call(SyscallOp.SYS_KEYS, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    # HTTP

    def http(self, url: str, method: str = "GET", headers: dict = None,
             body: str = None, timeout: int = 30,
             async_: bool = False, request_id: int = None) -> dict:
        """Make an HTTP request."""
        import json
        payload = {
            "url": url,
            "method": method,
            "timeout": timeout,
            "async": async_
        }

        if headers:
            payload["headers"] = headers
        if body:
            payload["body"] = body
        if request_id is not None:
            payload["request_id"] = request_id

        response = self.call(SyscallOp.SYS_HTTP, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "body": "", "error": response.payload_str}
        return {"success": False, "body": "", "error": "No response from kernel"}

    # Events (Pub/Sub)

    def subscribe(self, event_types: list) -> dict:
        """Subscribe to kernel events."""
        import json
        payload = {"event_types": event_types}

        response = self.call(SyscallOp.SYS_SUBSCRIBE, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def unsubscribe(self, event_types: list) -> dict:
        """Unsubscribe from kernel events."""
        import json
        payload = {"event_types": event_types}

        response = self.call(SyscallOp.SYS_UNSUBSCRIBE, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def poll_events(self, max_events: int = 10) -> dict:
        """Poll for pending events."""
        import json
        payload = {"max": max_events}

        response = self.call(SyscallOp.SYS_POLL_EVENTS, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "events": [], "count": 0, "error": response.payload_str}
        return {"success": False, "events": [], "count": 0, "error": "No response from kernel"}

    # Async Results

    def poll_async(self, max_results: int = 10) -> dict:
        """Poll for completed async syscall results."""
        import json
        payload = {"max": max_results}

        response = self.call(SyscallOp.SYS_ASYNC_POLL, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "results": [], "count": 0, "error": response.payload_str}
        return {"success": False, "results": [], "count": 0, "error": "No response from kernel"}

    def emit_event(self, event_type: str, data: dict = None) -> dict:
        """Emit a custom event to all subscribers."""
        import json
        payload = {
            "event_type": event_type,
            "data": data or {}
        }

        response = self.call(SyscallOp.SYS_EMIT, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    # World Simulation

    def world_create(self, name: str, config: dict = None) -> dict:
        """Create a new simulated world."""
        import json
        payload = {
            "name": name,
            "config": config or {}
        }

        response = self.call(SyscallOp.SYS_WORLD_CREATE, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def world_destroy(self, world_id: str, force: bool = False) -> dict:
        """Destroy a world."""
        import json
        payload = {"world_id": world_id, "force": force}

        response = self.call(SyscallOp.SYS_WORLD_DESTROY, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def world_list(self) -> dict:
        """List all active worlds."""
        import json

        response = self.call(SyscallOp.SYS_WORLD_LIST, "{}")
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "worlds": [], "count": 0, "error": response.payload_str}
        return {"success": False, "worlds": [], "count": 0, "error": "No response from kernel"}

    def world_join(self, world_id: str) -> dict:
        """Join a world."""
        import json
        payload = {"world_id": world_id}

        response = self.call(SyscallOp.SYS_WORLD_JOIN, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def world_leave(self) -> dict:
        """Leave the current world."""
        import json

        response = self.call(SyscallOp.SYS_WORLD_LEAVE, "{}")
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def world_event(self, world_id: str, event_type: str, params: dict = None) -> dict:
        """Inject a chaos event into a world."""
        import json
        payload = {
            "world_id": world_id,
            "event_type": event_type,
            "params": params or {}
        }

        response = self.call(SyscallOp.SYS_WORLD_EVENT, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def world_state(self, world_id: str) -> dict:
        """Get the current state and metrics of a world."""
        import json
        payload = {"world_id": world_id}

        response = self.call(SyscallOp.SYS_WORLD_STATE, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def world_snapshot(self, world_id: str) -> dict:
        """Create a snapshot of a world's state."""
        import json
        payload = {"world_id": world_id}

        response = self.call(SyscallOp.SYS_WORLD_SNAPSHOT, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def world_restore(self, snapshot: dict, new_world_id: str = None) -> dict:
        """Restore a world from a snapshot."""
        import json
        payload = {
            "snapshot": snapshot,
            "new_world_id": new_world_id or ""
        }

        response = self.call(SyscallOp.SYS_WORLD_RESTORE, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    # Tunnel (Remote Connectivity)

    def tunnel_connect(self, relay_url: str, machine_id: str = None,
                      token: str = None) -> dict:
        """Connect the kernel to a relay server for remote agent access."""
        import json
        payload = {"relay_url": relay_url}
        if machine_id:
            payload["machine_id"] = machine_id
        if token:
            payload["token"] = token

        response = self.call(SyscallOp.SYS_TUNNEL_CONNECT, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def tunnel_disconnect(self) -> dict:
        """Disconnect the kernel from the relay server."""
        import json

        response = self.call(SyscallOp.SYS_TUNNEL_DISCONNECT, "{}")
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def tunnel_status(self) -> dict:
        """Get the current tunnel connection status."""
        import json

        response = self.call(SyscallOp.SYS_TUNNEL_STATUS, "{}")
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def tunnel_list_remotes(self) -> dict:
        """List remote agents currently connected through the tunnel."""
        import json

        response = self.call(SyscallOp.SYS_TUNNEL_LIST_REMOTES, "{}")
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "agents": [], "error": response.payload_str}
        return {"success": False, "agents": [], "error": "No response from kernel"}

    def tunnel_config(self, relay_url: str = None, machine_id: str = None,
                     token: str = None, reconnect_interval: int = None) -> dict:
        """Configure tunnel settings without connecting."""
        import json
        payload = {}
        if relay_url:
            payload["relay_url"] = relay_url
        if machine_id:
            payload["machine_id"] = machine_id
        if token:
            payload["token"] = token
        if reconnect_interval is not None:
            payload["reconnect_interval"] = reconnect_interval

        response = self.call(SyscallOp.SYS_TUNNEL_CONFIG, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    # Metrics

    def get_system_metrics(self) -> dict:
        """Get system-wide metrics (CPU, memory, disk, network)."""
        import json

        response = self.call(SyscallOp.SYS_METRICS_SYSTEM, "{}")
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def get_agent_metrics(self, agent_id: int = None) -> dict:
        """Get metrics for a specific agent."""
        import json
        payload = {}
        if agent_id is not None:
            payload["agent_id"] = agent_id

        response = self.call(SyscallOp.SYS_METRICS_AGENT, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def get_all_agent_metrics(self) -> dict:
        """Get metrics for all running agents."""
        import json

        response = self.call(SyscallOp.SYS_METRICS_ALL_AGENTS, "{}")
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "agents": [], "error": response.payload_str}
        return {"success": False, "agents": [], "error": "No response from kernel"}

    def get_cgroup_metrics(self, cgroup_path: str = None) -> dict:
        """Get cgroup metrics for a sandboxed process."""
        import json
        payload = {}
        if cgroup_path:
            payload["cgroup_path"] = cgroup_path

        response = self.call(SyscallOp.SYS_METRICS_CGROUP, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    # ========== Audit Logging ==========

    def get_audit_log(self, category: str = None, agent_id: int = None,
                      since_id: int = 0, limit: int = 100) -> dict:
        """Get audit log entries with optional filtering.

        Args:
            category: Filter by category (SECURITY, AGENT_LIFECYCLE, IPC, etc.)
            agent_id: Filter by agent ID
            since_id: Get entries after this ID
            limit: Maximum entries to return (default 100)

        Returns:
            Dict with success, count, and entries list
        """
        import json
        payload = {"limit": limit}
        if category:
            payload["category"] = category
        if agent_id:
            payload["agent_id"] = agent_id
        if since_id:
            payload["since_id"] = since_id

        response = self.call(SyscallOp.SYS_GET_AUDIT_LOG, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def set_audit_config(self, **kwargs) -> dict:
        """Configure audit logging.

        Args:
            max_entries: Maximum entries to keep in memory
            log_syscalls: Log all syscalls (verbose)
            log_security: Log security events
            log_lifecycle: Log agent lifecycle events
            log_ipc: Log IPC events
            log_state: Log state store events
            log_resource: Log resource events
            log_network: Log network events
            log_world: Log world simulation events

        Returns:
            Dict with success and current config
        """
        import json
        response = self.call(SyscallOp.SYS_SET_AUDIT_CONFIG, json.dumps(kwargs))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    # ========== Execution Recording & Replay ==========

    def start_recording(self, include_think: bool = False, include_http: bool = False,
                        include_exec: bool = False, filter_agents: list = None,
                        max_entries: int = 50000) -> dict:
        """Start recording syscall execution for later replay.

        Args:
            include_think: Include LLM calls (non-deterministic)
            include_http: Include HTTP calls (non-deterministic)
            include_exec: Include exec calls (may be non-deterministic)
            filter_agents: Only record these agent IDs (empty = all)
            max_entries: Maximum entries to keep in buffer

        Returns:
            Dict with success status
        """
        import json
        payload = {
            "include_think": include_think,
            "include_http": include_http,
            "include_exec": include_exec,
            "max_entries": max_entries
        }
        if filter_agents:
            payload["filter_agents"] = filter_agents

        response = self.call(SyscallOp.SYS_RECORD_START, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def stop_recording(self) -> dict:
        """Stop recording syscall execution.

        Returns:
            Dict with success status and entry count
        """
        import json

        response = self.call(SyscallOp.SYS_RECORD_STOP, "{}")
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def get_recording_status(self, export: bool = False) -> dict:
        """Get current recording status and optionally export the recording.

        Args:
            export: If True, include the full recording data in response

        Returns:
            Dict with recording state, entry count, and optionally recording data
        """
        import json
        payload = {"export": export}

        response = self.call(SyscallOp.SYS_RECORD_STATUS, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def start_replay(self, recording_data: str) -> dict:
        """Start replaying a recorded execution session.

        Args:
            recording_data: JSON string of recorded execution entries

        Returns:
            Dict with success status and total entries to replay
        """
        import json
        payload = {"recording": recording_data}

        response = self.call(SyscallOp.SYS_REPLAY_START, json.dumps(payload))
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    def get_replay_status(self) -> dict:
        """Get current replay status and progress.

        Returns:
            Dict with replay state, progress, entries replayed/skipped, and errors
        """
        import json

        response = self.call(SyscallOp.SYS_REPLAY_STATUS, "{}")
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"success": False, "error": response.payload_str}
        return {"success": False, "error": "No response from kernel"}

    # Convenience Aliases

    def read(self, path: str) -> str:
        """Alias for read_file that returns just the content string."""
        result = self.read_file(path)
        if result.get("success"):
            return result.get("content", "")
        raise IOError(result.get("error", "Read failed"))

    def write(self, path: str, content: str, mode: str = "write") -> dict:
        """Alias for write_file."""
        return self.write_file(path, content, mode)

    def register(self, name: str) -> dict:
        """Alias for register_name."""
        return self.register_name(name)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# Backwards compatibility alias
AgentOSClient = CloveClient


# Convenience function for quick testing
def connect(socket_path: str = '/tmp/clove.sock') -> CloveClient:
    """Create and connect a client"""
    client = CloveClient(socket_path)
    if not client.connect():
        raise ConnectionError(f"Failed to connect to {socket_path}")
    return client
