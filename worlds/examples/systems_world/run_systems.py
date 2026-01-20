#!/usr/bin/env python3
"""
Systems World Simulation Runner

Simulates a production infrastructure with servers, SRE agents,
and chaos injection.

Usage:
    # Local (kernel must be running)
    python run_systems.py

    # With LLM-powered SRE
    python run_systems.py --use-llm

    # High chaos mode
    python run_systems.py --chaos-level high

    # Via CLI (remote)
    agentos agent run worlds/examples/systems_world/run_systems.py --machine <machine_id>
"""

import sys
import os
import time
import argparse
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'agents', 'python_sdk'))
from agentos import AgentOSClient

from systems_config import SYSTEMS_CONFIG, SRE_SKILL_LEVELS


def run_simulation(chaos_level: str = "medium", use_llm: bool = False, verbose: bool = True):
    """Run the systems simulation."""

    chaos_multiplier = {
        "low": 0.5,
        "medium": 1.0,
        "high": 2.0,
        "extreme": 3.0,
    }.get(chaos_level, 1.0)

    print("=" * 70)
    print("SYSTEMS WORLD SIMULATION")
    print(f"Infrastructure: {SYSTEMS_CONFIG['name']}")
    print(f"Servers: {len(SYSTEMS_CONFIG['servers'])}")
    print(f"Rounds: {SYSTEMS_CONFIG['rounds']}")
    print(f"Chaos Level: {chaos_level} ({chaos_multiplier}x)")
    print(f"LLM Mode: {'ON' if use_llm else 'OFF'}")
    print("=" * 70)

    with AgentOSClient() as client:
        spawned_agents = []
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Spawn servers
        print("\nSpawning servers...")
        for server in SYSTEMS_CONFIG["servers"]:
            server_script = os.path.join(script_dir, "server_agent.py")

            # Create wrapper
            wrapper_code = f'''
import sys
sys.argv = ["server_agent.py", "--name", "{server['name']}"]
exec(open("{server_script}").read())
'''
            wrapper_path = f"/tmp/server_{server['name']}.py"
            with open(wrapper_path, "w") as f:
                f.write(wrapper_code)

            result = client.spawn(
                name=f"server-{server['name']}",
                script=wrapper_path,
                sandboxed=False
            )

            if result.get("success"):
                spawned_agents.append(f"server-{server['name']}")
                print(f"  [{server['type'].upper()}] {server['name']}: PID {result.get('pid')}")
            else:
                print(f"  FAILED {server['name']}: {result.get('error')}")

        time.sleep(2)

        # Spawn SRE agent
        print("\nSpawning SRE...")
        sre_script = os.path.join(script_dir, "sre_agent.py")

        sre_wrapper = f'''
import sys
sys.argv = ["sre_agent.py", "--name", "oncall", "--skill", "mid"{', "--use-llm"' if use_llm else ''}]
exec(open("{sre_script}").read())
'''
        wrapper_path = "/tmp/sre_oncall.py"
        with open(wrapper_path, "w") as f:
            f.write(sre_wrapper)

        result = client.spawn(
            name="sre-oncall",
            script=wrapper_path,
            sandboxed=False
        )

        if result.get("success"):
            spawned_agents.append("sre-oncall")
            print(f"  SRE oncall: PID {result.get('pid')}")
        else:
            print(f"  FAILED to spawn SRE: {result.get('error')}")

        time.sleep(1)

        # Run simulation
        print("\n" + "=" * 70)
        print("SIMULATION RUNNING")
        print("=" * 70)

        total_uptime = 0
        total_checks = 0

        for round_num in range(1, SYSTEMS_CONFIG["rounds"] + 1):
            print(f"\n--- Round {round_num} ---")

            # Generate traffic
            base_traffic = SYSTEMS_CONFIG["base_traffic"]
            traffic = int(base_traffic * random.uniform(0.8, 1.5))

            # Send traffic update to all servers
            for server in SYSTEMS_CONFIG["servers"]:
                client.send_message({
                    "type": "traffic_update",
                    "traffic": traffic // len(SYSTEMS_CONFIG["servers"]),
                    "round": round_num
                }, to_name=f"server-{server['name']}")

            # Maybe inject chaos
            for scenario in SYSTEMS_CONFIG["chaos_scenarios"]:
                adjusted_prob = scenario["probability"] * chaos_multiplier
                if random.random() < adjusted_prob:
                    # Pick a random server
                    target_server = random.choice(SYSTEMS_CONFIG["servers"])

                    print(f"  CHAOS: {scenario['name']} -> {target_server['name']}")

                    client.send_message({
                        "type": "chaos_inject",
                        "chaos": scenario
                    }, to_name=f"server-{target_server['name']}")

            # Trigger SRE monitoring round
            time.sleep(0.5)
            client.send_message({
                "type": "round_tick",
                "round": round_num
            }, to_name="sre-oncall")

            # Check system health
            time.sleep(1)
            healthy_servers = 0
            for server in SYSTEMS_CONFIG["servers"]:
                result = client.fetch(f"metrics:{server['name']}")
                if result.get("success") and result.get("exists"):
                    metrics = result["value"]
                    health = metrics.get("health", 0)
                    if health > 50:
                        healthy_servers += 1

                    if verbose:
                        status = "OK" if health > 80 else "WARN" if health > 50 else "CRIT"
                        print(f"    {server['name']}: {status} (health={health}, cpu={metrics.get('cpu_usage', 0):.0f}%)")

            # Track uptime
            total_checks += len(SYSTEMS_CONFIG["servers"])
            total_uptime += healthy_servers

            time.sleep(1)

        # Calculate SLA
        sla = (total_uptime / total_checks) * 100 if total_checks > 0 else 0

        # Final report
        print("\n" + "=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)

        print(f"\nInfrastructure Summary:")
        print(f"  Rounds: {SYSTEMS_CONFIG['rounds']}")
        print(f"  Chaos Level: {chaos_level}")
        print(f"  Availability: {sla:.2f}%")
        print(f"  SLA Target: {SYSTEMS_CONFIG['sla_target']}%")

        if sla >= SYSTEMS_CONFIG["sla_target"]:
            print(f"\n  SLA MET! Infrastructure remained stable.")
        else:
            print(f"\n  SLA MISSED by {SYSTEMS_CONFIG['sla_target'] - sla:.2f}%")

        # Get SRE stats
        # (In a real implementation, we'd fetch this from the SRE agent)

        # Cleanup
        print("\nShutting down...")

        # Send shutdown to all agents
        for agent_name in spawned_agents:
            try:
                client.send_message({"type": "shutdown"}, to_name=agent_name)
            except:
                pass

        time.sleep(2)

        # Kill any remaining
        for agent_name in spawned_agents:
            try:
                client.kill(name=agent_name)
            except:
                pass

        print("Simulation complete!")

        return sla


def main():
    parser = argparse.ArgumentParser(description="Run Systems World Simulation")
    parser.add_argument("--chaos-level", default="medium",
                       choices=["low", "medium", "high", "extreme"],
                       help="Chaos injection intensity")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM for SRE diagnosis")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    args = parser.parse_args()

    run_simulation(args.chaos_level, args.use_llm, verbose=not args.quiet)


if __name__ == "__main__":
    main()
