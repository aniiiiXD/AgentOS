#!/usr/bin/env python3
"""
Server Agent - Systems World

Simulates a server that reports metrics, can experience failures,
and responds to SRE remediation commands.
"""

import sys
import os
import json
import random
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'agents', 'python_sdk'))
from agentos import AgentOS

from systems_config import SYSTEMS_CONFIG


class ServerAgent:
    def __init__(self, server_config: dict):
        self.config = server_config
        self.name = server_config["name"]

        self.agent = AgentOS(f"server-{self.name}")
        self.agent.register_name(f"server-{self.name}")

        # Server state
        self.health = 100
        self.cpu_usage = random.uniform(10, 30)
        self.memory_usage = random.uniform(20, 40)
        self.disk_usage = random.uniform(30, 50)
        self.request_rate = 0
        self.error_rate = 0
        self.latency_p50 = random.uniform(10, 50)
        self.latency_p99 = random.uniform(50, 200)

        # Active chaos effects
        self.chaos_effects = []
        self.isolated = False
        self.services_down = set()

        self.running = True
        self.round = 0

        self.agent.write(f"[{self.name}] Server online")
        self.agent.write(f"  Type: {self.config['type']}")
        self.agent.write(f"  Services: {', '.join(self.config['services'])}")

    def apply_chaos(self, chaos: dict):
        """Apply a chaos effect to this server."""
        effect = chaos.get("effect", {})

        if "cpu_usage" in effect:
            self.cpu_usage = effect["cpu_usage"]
            self.agent.write(f"[{self.name}] CHAOS: CPU spike to {self.cpu_usage}%")

        if "memory_growth" in effect:
            growth = effect["memory_growth"]
            self.memory_usage = min(100, self.memory_usage + growth)
            self.agent.write(f"[{self.name}] CHAOS: Memory leak, now at {self.memory_usage:.0f}%")

        if "isolated" in effect:
            self.isolated = True
            self.agent.write(f"[{self.name}] CHAOS: Network partition!")

        if "disk_usage" in effect:
            self.disk_usage = effect["disk_usage"]
            self.agent.write(f"[{self.name}] CHAOS: Disk at {self.disk_usage}%")

        if "service_down" in effect:
            if self.config["services"]:
                crashed = random.choice(self.config["services"])
                self.services_down.add(crashed)
                self.agent.write(f"[{self.name}] CHAOS: Service {crashed} CRASHED!")

        if "traffic_multiplier" in effect:
            self.request_rate *= effect["traffic_multiplier"]
            self.cpu_usage = min(100, self.cpu_usage * 1.5)
            self.agent.write(f"[{self.name}] CHAOS: Traffic surge!")

        if "db_latency_multiplier" in effect and self.config["type"] == "database":
            self.latency_p99 *= effect["db_latency_multiplier"]
            self.agent.write(f"[{self.name}] CHAOS: Database slowdown, p99={self.latency_p99:.0f}ms")

        self.chaos_effects.append(chaos)

    def handle_remediation(self, action: str) -> bool:
        """Handle a remediation action from SRE."""
        self.agent.write(f"[{self.name}] Remediation: {action}")

        if action == "restart_service":
            if self.services_down:
                service = self.services_down.pop()
                self.agent.write(f"[{self.name}] Restarted {service}")
                return True
            return False

        elif action == "clear_cache":
            old_mem = self.memory_usage
            self.memory_usage = max(20, self.memory_usage - 30)
            self.agent.write(f"[{self.name}] Cleared cache: memory {old_mem:.0f}% -> {self.memory_usage:.0f}%")
            return True

        elif action == "failover":
            if self.isolated:
                self.isolated = False
                self.agent.write(f"[{self.name}] Network restored via failover")
                return True
            return False

        elif action == "rate_limit":
            self.request_rate = max(100, self.request_rate * 0.3)
            self.cpu_usage = max(20, self.cpu_usage * 0.6)
            self.agent.write(f"[{self.name}] Rate limiting enabled")
            return True

        elif action == "scale_up":
            self.cpu_usage = max(10, self.cpu_usage * 0.5)
            self.agent.write(f"[{self.name}] Scaled up, load distributed")
            return True

        elif action == "rollback":
            # Rollback clears most chaos effects
            self.chaos_effects.clear()
            self.cpu_usage = random.uniform(20, 40)
            self.latency_p99 = random.uniform(50, 200)
            self.agent.write(f"[{self.name}] Rolled back to stable version")
            return True

        return False

    def simulate_round(self, traffic: int):
        """Simulate one round of server operation."""
        self.round += 1
        self.request_rate = traffic

        # Natural drift
        self.cpu_usage += random.uniform(-5, 5)
        self.cpu_usage = max(5, min(100, self.cpu_usage))

        self.memory_usage += random.uniform(-2, 3)
        self.memory_usage = max(10, min(100, self.memory_usage))

        self.disk_usage += random.uniform(0, 0.5)
        self.disk_usage = min(100, self.disk_usage)

        # Calculate health
        self.health = 100
        if self.cpu_usage > 90:
            self.health -= 30
        elif self.cpu_usage > 70:
            self.health -= 10

        if self.memory_usage > 95:
            self.health -= 40
        elif self.memory_usage > 80:
            self.health -= 15

        if self.services_down:
            self.health -= 50

        if self.isolated:
            self.health = 0

        self.health = max(0, self.health)

        # Error rate based on health
        if self.health < 50:
            self.error_rate = random.uniform(5, 20)
        elif self.health < 80:
            self.error_rate = random.uniform(1, 5)
        else:
            self.error_rate = random.uniform(0, 1)

        # Latency
        if self.cpu_usage > 80:
            self.latency_p99 = random.uniform(500, 2000)
        else:
            self.latency_p99 = random.uniform(50, 300)

    def get_metrics(self) -> dict:
        """Return current server metrics."""
        return {
            "server": self.name,
            "type": self.config["type"],
            "round": self.round,
            "health": self.health,
            "cpu_usage": round(self.cpu_usage, 1),
            "memory_usage": round(self.memory_usage, 1),
            "disk_usage": round(self.disk_usage, 1),
            "request_rate": self.request_rate,
            "error_rate": round(self.error_rate, 2),
            "latency_p50": round(self.latency_p50, 0),
            "latency_p99": round(self.latency_p99, 0),
            "services_up": [s for s in self.config["services"] if s not in self.services_down],
            "services_down": list(self.services_down),
            "isolated": self.isolated,
            "chaos_active": len(self.chaos_effects) > 0,
        }

    def publish_metrics(self):
        """Publish metrics to state store."""
        metrics = self.get_metrics()
        self.agent.store(f"metrics:{self.name}", metrics, scope="global", ttl=30)

    def handle_message(self, msg: dict):
        """Handle incoming messages."""
        content = msg.get("message", {})
        msg_type = content.get("type")

        if msg_type == "chaos_inject":
            chaos = content.get("chaos")
            if chaos:
                self.apply_chaos(chaos)

        elif msg_type == "remediate":
            action = content.get("action")
            sender = msg.get("from_name")
            success = self.handle_remediation(action)

            # Report result
            self.agent.send_message({
                "type": "remediation_result",
                "server": self.name,
                "action": action,
                "success": success
            }, to_name=sender)

        elif msg_type == "traffic_update":
            traffic = content.get("traffic", 1000)
            self.simulate_round(traffic)
            self.publish_metrics()

        elif msg_type == "shutdown":
            self.agent.write(f"[{self.name}] Shutting down...")
            self.running = False

    def run(self):
        """Main server loop."""
        while self.running:
            messages = self.agent.recv_messages(max_messages=10)

            if messages.get("success") and messages.get("messages"):
                for msg in messages["messages"]:
                    self.handle_message(msg)

            # Periodically publish metrics
            self.publish_metrics()

            time.sleep(0.5)

        self.agent.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Server Agent")
    parser.add_argument("--name", required=True, help="Server name from config")
    args = parser.parse_args()

    # Find server config
    server_config = None
    for server in SYSTEMS_CONFIG["servers"]:
        if server["name"] == args.name:
            server_config = server
            break

    if not server_config:
        print(f"Unknown server: {args.name}")
        sys.exit(1)

    server = ServerAgent(server_config)
    server.run()


if __name__ == "__main__":
    main()
