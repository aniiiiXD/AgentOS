#!/usr/bin/env python3
"""
AgentOS Python SDK

Client library for communicating with the AgentOS kernel via Unix domain sockets.
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
    SYS_EXIT = 0xFF   # Graceful shutdown


@dataclass
class Message:
    """AgentOS message"""
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


class AgentOSClient:
    """Client for communicating with AgentOS kernel"""

    def __init__(self, socket_path: str = '/tmp/agentos.sock'):
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None
        self._agent_id = 0

    def connect(self) -> bool:
        """Connect to the AgentOS kernel"""
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
    def echo(self, message: str) -> Optional[str]:
        """Echo a message (for testing)"""
        response = self.call(SyscallOp.SYS_NOOP, message)
        return response.payload_str if response else None

    def think(self, prompt: str) -> dict:
        """Send a prompt to the LLM via Gemini API.

        Returns dict with 'content', 'tokens', and optionally 'error'.
        """
        import json
        response = self.call(SyscallOp.SYS_THINK, prompt)
        if response:
            try:
                return json.loads(response.payload_str)
            except json.JSONDecodeError:
                return {"content": response.payload_str, "error": None}
        return {"content": "", "error": "No response from kernel"}

    def exit(self) -> bool:
        """Request graceful exit"""
        response = self.call(SyscallOp.SYS_EXIT)
        return response is not None

    def spawn(self, name: str, script: str, sandboxed: bool = True,
              network: bool = False, limits: dict = None) -> Optional[dict]:
        """Spawn a new sandboxed agent"""
        import json
        payload = {
            "name": name,
            "script": script,
            "sandboxed": sandboxed,
            "network": network
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

    def list_agents(self) -> list:
        """List all running agents"""
        import json
        response = self.call(SyscallOp.SYS_LIST)
        if response:
            return json.loads(response.payload_str)
        return []

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# Convenience function for quick testing
def connect(socket_path: str = '/tmp/agentos.sock') -> AgentOSClient:
    """Create and connect a client"""
    client = AgentOSClient(socket_path)
    if not client.connect():
        raise ConnectionError(f"Failed to connect to {socket_path}")
    return client


if __name__ == '__main__':
    # Quick test
    print("AgentOS Python SDK")
    print("==================")
    print(f"Socket path: /tmp/agentos.sock")
    print(f"Header size: {HEADER_SIZE} bytes")
    print(f"Magic bytes: 0x{MAGIC_BYTES:08X}")
    print()
    print("Usage:")
    print("  from agentos import AgentOSClient")
    print("  ")
    print("  with AgentOSClient() as client:")
    print("      response = client.echo('Hello!')")
    print("      print(response)")
