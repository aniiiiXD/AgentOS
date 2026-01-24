#!/usr/bin/env python3
"""
Clove Fleet Client

High-level Python client for managing Clove fleets.
Combines relay API access with agent execution.
"""

import os
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Machine:
    """Represents a machine in the fleet."""
    machine_id: str
    provider: str
    status: str
    ip_address: str = ""
    created_at: str = ""
    last_seen: str = ""
    metadata: Dict[str, Any] = None

    def is_connected(self) -> bool:
        return self.status == "connected"


@dataclass
class Agent:
    """Represents a running agent."""
    agent_id: int
    agent_name: str
    target_machine: str
    status: str
    syscalls_sent: int = 0


class FleetClientError(Exception):
    """Fleet client error."""
    pass


class FleetClient:
    """Client for managing Clove fleets."""

    def __init__(self, relay_url: str = None, api_token: str = None):
        self.relay_url = relay_url or os.environ.get("RELAY_API_URL", "http://localhost:8766")
        self.api_token = api_token or os.environ.get("CLOVE_API_TOKEN", "")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        session = await self._get_session()
        url = f"{self.relay_url.rstrip('/')}{endpoint}"

        try:
            async with session.request(method, url, json=data) as resp:
                body = await resp.json()
                if resp.status >= 400:
                    raise FleetClientError(body.get("error", f"HTTP {resp.status}"))
                return body
        except aiohttp.ClientError as e:
            raise FleetClientError(f"Connection error: {e}")

    async def list_machines(self) -> List[Machine]:
        data = await self._request("GET", "/api/v1/machines")
        return [Machine(**m) for m in data.get("machines", [])]

    async def get_machine(self, machine_id: str) -> Machine:
        data = await self._request("GET", f"/api/v1/machines/{machine_id}")
        return Machine(**data)

    async def get_connected_machines(self) -> List[Machine]:
        machines = await self.list_machines()
        return [m for m in machines if m.is_connected()]

    async def list_agents(self, machine_id: str = None) -> List[Agent]:
        endpoint = "/api/v1/agents"
        if machine_id:
            endpoint += f"?machine_id={machine_id}"
        data = await self._request("GET", endpoint)
        return [Agent(**a) for a in data.get("agents", [])]

    async def deploy_agent(self, script_path: str, machine_id: str,
                          args: List[str] = None) -> Dict[str, Any]:
        path = Path(script_path)
        if not path.exists():
            raise FleetClientError(f"Script not found: {script_path}")

        script_content = path.read_text()

        return await self._request("POST", "/api/v1/agents/deploy", {
            "machine_id": machine_id,
            "script_content": script_content,
            "script_name": path.name,
            "args": args or []
        })

    async def run_on_all(self, script_path: str, args: List[str] = None,
                        filter_fn: Callable[[Machine], bool] = None) -> List[Dict]:
        machines = await self.get_connected_machines()
        if filter_fn:
            machines = [m for m in machines if filter_fn(m)]
        if not machines:
            raise FleetClientError("No machines available")

        results = []
        for machine in machines:
            try:
                result = await self.deploy_agent(script_path, machine.machine_id, args)
                results.append({"machine_id": machine.machine_id, "status": "deployed", **result})
            except FleetClientError as e:
                results.append({"machine_id": machine.machine_id, "status": "failed", "error": str(e)})
        return results

    async def stop_agent(self, machine_id: str, agent_id: int) -> bool:
        await self._request("POST", f"/api/v1/agents/{agent_id}/stop", {"machine_id": machine_id})
        return True

    async def get_status(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/status")

    async def health_check(self) -> bool:
        try:
            data = await self._request("GET", "/api/v1/health")
            return data.get("status") == "healthy"
        except FleetClientError:
            return False


class SyncFleetClient:
    """Synchronous wrapper for FleetClient."""

    def __init__(self, relay_url: str = None, api_token: str = None):
        self.relay_url = relay_url
        self.api_token = api_token

    def _run(self, coro):
        async def _wrapper():
            client = FleetClient(self.relay_url, self.api_token)
            try:
                return await coro(client)
            finally:
                await client.close()

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_wrapper())
        finally:
            loop.close()

    def list_machines(self) -> List[Machine]:
        return self._run(lambda c: c.list_machines())

    def get_machine(self, machine_id: str) -> Machine:
        return self._run(lambda c: c.get_machine(machine_id))

    def get_connected_machines(self) -> List[Machine]:
        return self._run(lambda c: c.get_connected_machines())

    def list_agents(self, machine_id: str = None) -> List[Agent]:
        return self._run(lambda c: c.list_agents(machine_id))

    def deploy_agent(self, script_path: str, machine_id: str,
                    args: List[str] = None) -> Dict[str, Any]:
        return self._run(lambda c: c.deploy_agent(script_path, machine_id, args))

    def run_on_all(self, script_path: str, args: List[str] = None) -> List[Dict]:
        return self._run(lambda c: c.run_on_all(script_path, args))

    def stop_agent(self, machine_id: str, agent_id: int) -> bool:
        return self._run(lambda c: c.stop_agent(machine_id, agent_id))

    def get_status(self) -> Dict[str, Any]:
        return self._run(lambda c: c.get_status())

    def health_check(self) -> bool:
        return self._run(lambda c: c.health_check())
