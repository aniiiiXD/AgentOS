#!/usr/bin/env python3
"""
SRE Agent - Systems World

An SRE (Site Reliability Engineer) agent that monitors servers,
detects issues, and performs remediation. Can use LLM for diagnosis.
"""

import sys
import os
import json
import random
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'agents', 'python_sdk'))
from agentos import AgentOS

from systems_config import SYSTEMS_CONFIG, SRE_SKILL_LEVELS


class SREAgent:
    def __init__(self, name: str, skill_level: str = "mid", use_llm: bool = False):
        self.name = name
        self.skill = SRE_SKILL_LEVELS.get(skill_level, SRE_SKILL_LEVELS["mid"])
        self.skill_level = skill_level
        self.use_llm = use_llm

        self.agent = AgentOS(f"sre-{name}")
        self.agent.register_name(f"sre-{name}")

        # Tracking
        self.alerts = []
        self.active_incidents = {}
        self.remediation_cooldowns = {}
        self.stats = {
            "alerts_handled": 0,
            "incidents_resolved": 0,
            "failed_remediations": 0,
            "mttr_total": 0,  # mean time to resolve
        }

        self.running = True
        self.round = 0

        self.agent.write(f"[SRE {self.name}] Online!")
        self.agent.write(f"  Skill Level: {skill_level}")
        self.agent.write(f"  Diagnosis Accuracy: {self.skill['diagnosis_accuracy']:.0%}")
        self.agent.write(f"  LLM Assist: {'ON' if use_llm else 'OFF'}")

    def collect_metrics(self) -> list:
        """Collect metrics from all servers."""
        metrics = []
        for server in SYSTEMS_CONFIG["servers"]:
            result = self.agent.fetch(f"metrics:{server['name']}")
            if result.get("success") and result.get("exists"):
                metrics.append(result["value"])
        return metrics

    def check_alerts(self, metrics: list) -> list:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        thresholds = SYSTEMS_CONFIG["alert_thresholds"]

        for m in metrics:
            server = m.get("server")

            # Check CPU
            cpu = m.get("cpu_usage", 0)
            if cpu > thresholds["cpu_usage"]["critical"]:
                alerts.append({"server": server, "metric": "cpu_usage", "level": "critical", "value": cpu})
            elif cpu > thresholds["cpu_usage"]["warning"]:
                alerts.append({"server": server, "metric": "cpu_usage", "level": "warning", "value": cpu})

            # Check Memory
            mem = m.get("memory_usage", 0)
            if mem > thresholds["memory_usage"]["critical"]:
                alerts.append({"server": server, "metric": "memory_usage", "level": "critical", "value": mem})
            elif mem > thresholds["memory_usage"]["warning"]:
                alerts.append({"server": server, "metric": "memory_usage", "level": "warning", "value": mem})

            # Check Error Rate
            err = m.get("error_rate", 0)
            if err > thresholds["error_rate"]["critical"]:
                alerts.append({"server": server, "metric": "error_rate", "level": "critical", "value": err})
            elif err > thresholds["error_rate"]["warning"]:
                alerts.append({"server": server, "metric": "error_rate", "level": "warning", "value": err})

            # Check Latency
            lat = m.get("latency_p99", 0)
            if lat > thresholds["latency_p99"]["critical"]:
                alerts.append({"server": server, "metric": "latency_p99", "level": "critical", "value": lat})
            elif lat > thresholds["latency_p99"]["warning"]:
                alerts.append({"server": server, "metric": "latency_p99", "level": "warning", "value": lat})

            # Check services
            if m.get("services_down"):
                alerts.append({"server": server, "metric": "service_down", "level": "critical",
                              "value": m["services_down"]})

            # Check isolation
            if m.get("isolated"):
                alerts.append({"server": server, "metric": "network", "level": "critical", "value": "isolated"})

            # Check health
            health = m.get("health", 100)
            if health < 50:
                alerts.append({"server": server, "metric": "health", "level": "critical", "value": health})

        return alerts

    def diagnose_issue(self, alert: dict, server_metrics: dict) -> str:
        """Diagnose the issue and recommend remediation."""

        metric = alert.get("metric")
        value = alert.get("value")
        server = alert.get("server")

        # Use LLM for diagnosis if enabled
        if self.use_llm:
            prompt = f"""You are an SRE analyzing a production incident.

Server: {server}
Alert: {metric} is at {value}
Full metrics: {json.dumps(server_metrics, indent=2)}

Available remediation actions:
{json.dumps([a['name'] + ': ' + a['description'] for a in SYSTEMS_CONFIG['remediation_actions']], indent=2)}

Recommend ONE action to take. Reply with just the action name."""

            result = self.agent.think(prompt)
            if result.get("success"):
                recommended = result.get("content", "").strip().lower()
                # Match to valid action
                for action in SYSTEMS_CONFIG["remediation_actions"]:
                    if action["name"] in recommended:
                        return action["name"]

        # Rule-based diagnosis
        if metric == "cpu_usage":
            if value > 90:
                return "scale_up"
            else:
                return "rate_limit"

        elif metric == "memory_usage":
            return "clear_cache"

        elif metric == "service_down":
            return "restart_service"

        elif metric == "network":
            return "failover"

        elif metric == "latency_p99":
            if server_metrics.get("type") == "database":
                return "scale_up"
            return "restart_service"

        elif metric == "error_rate":
            return "rollback"

        elif metric == "health":
            return "restart_service"

        return "restart_service"  # Default

    def can_remediate(self, action: str) -> bool:
        """Check if action is off cooldown."""
        if action not in self.remediation_cooldowns:
            return True

        cooldown_until = self.remediation_cooldowns[action]
        return self.round >= cooldown_until

    def remediate(self, server: str, action: str) -> bool:
        """Send remediation command to server."""
        if not self.can_remediate(action):
            self.agent.write(f"[SRE {self.name}] {action} on cooldown")
            return False

        # Find action config
        action_config = None
        for a in SYSTEMS_CONFIG["remediation_actions"]:
            if a["name"] == action:
                action_config = a
                break

        if not action_config:
            return False

        # Apply skill-based success modifier
        base_success = action_config["success_rate"]
        modified_success = base_success * self.skill["remediation_speed"]

        self.agent.write(f"[SRE {self.name}] Executing {action} on {server}...")

        # Send to server
        self.agent.send_message({
            "type": "remediate",
            "action": action,
            "server": server
        }, to_name=f"server-{server}")

        # Set cooldown
        self.remediation_cooldowns[action] = self.round + action_config["cooldown"]

        return True

    def process_round(self):
        """Process one monitoring round."""
        self.round += 1

        # Collect metrics
        metrics = self.collect_metrics()
        if not metrics:
            return

        # Check for alerts
        alerts = self.check_alerts(metrics)

        # Check if overwhelmed
        critical_alerts = [a for a in alerts if a["level"] == "critical"]
        if len(critical_alerts) > self.skill["panic_threshold"]:
            self.agent.write(f"[SRE {self.name}] OVERWHELMED! {len(critical_alerts)} critical alerts!")
            # Prioritize highest impact
            alerts = sorted(alerts, key=lambda x: 1 if x["level"] == "critical" else 0, reverse=True)
            alerts = alerts[:self.skill["panic_threshold"]]

        # Handle alerts
        for alert in alerts:
            server = alert.get("server")
            level = alert.get("level")
            metric = alert.get("metric")
            value = alert.get("value")

            # Skip warnings if we have criticals
            if level == "warning" and critical_alerts:
                continue

            self.agent.write(f"[SRE {self.name}] ALERT [{level.upper()}] {server}: {metric}={value}")
            self.stats["alerts_handled"] += 1

            # Find server metrics
            server_metrics = next((m for m in metrics if m["server"] == server), {})

            # Diagnose (with skill-based accuracy)
            if random.random() < self.skill["diagnosis_accuracy"]:
                action = self.diagnose_issue(alert, server_metrics)
            else:
                # Wrong diagnosis
                actions = [a["name"] for a in SYSTEMS_CONFIG["remediation_actions"]]
                action = random.choice(actions)
                self.agent.write(f"[SRE {self.name}] (misdiagnosis)")

            # Attempt remediation
            if self.remediate(server, action):
                self.stats["incidents_resolved"] += 1
            else:
                self.stats["failed_remediations"] += 1

        # Report stats periodically
        if self.round % 5 == 0:
            self.agent.write(f"[SRE {self.name}] Stats: {self.stats['alerts_handled']} alerts, "
                           f"{self.stats['incidents_resolved']} resolved, "
                           f"{self.stats['failed_remediations']} failed")

    def handle_message(self, msg: dict):
        """Handle incoming messages."""
        content = msg.get("message", {})
        msg_type = content.get("type")

        if msg_type == "remediation_result":
            server = content.get("server")
            action = content.get("action")
            success = content.get("success")

            if success:
                self.agent.write(f"[SRE {self.name}] Remediation succeeded: {action} on {server}")
            else:
                self.agent.write(f"[SRE {self.name}] Remediation failed: {action} on {server}")
                self.stats["failed_remediations"] += 1

        elif msg_type == "round_tick":
            self.process_round()

        elif msg_type == "shutdown":
            self.agent.write(f"[SRE {self.name}] Shift over. Final stats: {self.stats}")
            self.running = False

    def run(self):
        """Main SRE loop."""
        while self.running:
            messages = self.agent.recv_messages(max_messages=10)

            if messages.get("success") and messages.get("messages"):
                for msg in messages["messages"]:
                    self.handle_message(msg)

            time.sleep(0.3)

        self.agent.exit(0)


def main():
    parser = argparse.ArgumentParser(description="SRE Agent")
    parser.add_argument("--name", required=True, help="SRE name")
    parser.add_argument("--skill", default="mid",
                       choices=list(SRE_SKILL_LEVELS.keys()),
                       help="Skill level")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM for diagnosis")
    args = parser.parse_args()

    sre = SREAgent(args.name, args.skill, args.use_llm)
    sre.run()


if __name__ == "__main__":
    main()
