#!/usr/bin/env python3
"""
Office World Simulation Runner

Spawns the manager and worker agents, runs the simulation,
and displays results.

Usage:
    # Local (kernel must be running)
    python run_office.py

    # With LLM-powered workers
    python run_office.py --use-llm

    # Custom number of workers
    python run_office.py --workers 3

    # Via CLI (remote)
    agentos agent run worlds/examples/office_world/run_office.py --machine <machine_id>
"""

import sys
import os
import time
import argparse
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'agents', 'python_sdk'))
from agentos import AgentOSClient

from office_config import OFFICE_CONFIG, WORKER_PERSONALITIES

# Worker names pool
WORKER_NAMES = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "henry"]


def run_simulation(num_workers: int = 5, use_llm: bool = False, verbose: bool = True):
    """Run the office simulation."""

    print("=" * 70)
    print("OFFICE WORLD SIMULATION")
    print(f"Office: {OFFICE_CONFIG['name']}")
    print(f"Workers: {num_workers}")
    print(f"Rounds: {OFFICE_CONFIG['rounds']}")
    print(f"LLM Mode: {'ON' if use_llm else 'OFF'}")
    print("=" * 70)

    with AgentOSClient() as client:
        # Clear any previous state
        client.store("registered_workers", [], scope="global")
        client.store("leaderboard", {}, scope="global")

        spawned_agents = []
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Spawn workers first
        print("\nSpawning workers...")
        worker_names = WORKER_NAMES[:num_workers]
        personalities = list(WORKER_PERSONALITIES.keys())

        for i, name in enumerate(worker_names):
            personality = personalities[i % len(personalities)]

            worker_script = os.path.join(script_dir, "worker_agent.py")
            cmd_args = f"--name {name} --personality {personality}"
            if use_llm:
                cmd_args += " --use-llm"

            # Create a wrapper script that calls worker with args
            wrapper_code = f'''
import sys
sys.argv = ["worker_agent.py", "--name", "{name}", "--personality", "{personality}"{', "--use-llm"' if use_llm else ''}]
exec(open("{worker_script}").read())
'''
            # Write temp wrapper
            wrapper_path = f"/tmp/worker_{name}.py"
            with open(wrapper_path, "w") as f:
                f.write(wrapper_code)

            result = client.spawn(
                name=f"worker-{name}",
                script=wrapper_path,
                sandboxed=False  # Need IPC access
            )

            if result.get("success"):
                spawned_agents.append(f"worker-{name}")
                print(f"  Spawned {name} ({personality}): PID {result.get('pid')}")
            else:
                print(f"  FAILED to spawn {name}: {result.get('error')}")

        # Wait for workers to register
        time.sleep(2)

        # Spawn manager
        print("\nSpawning manager...")
        manager_script = os.path.join(script_dir, "manager_agent.py")

        result = client.spawn(
            name="office-manager",
            script=manager_script,
            sandboxed=False
        )

        if result.get("success"):
            spawned_agents.append("office-manager")
            print(f"  Spawned manager: PID {result.get('pid')}")
        else:
            print(f"  FAILED to spawn manager: {result.get('error')}")
            # Cleanup workers
            for agent_name in spawned_agents:
                client.kill(name=agent_name)
            return

        # Monitor simulation
        print("\n" + "=" * 70)
        print("SIMULATION RUNNING")
        print("=" * 70)

        simulation_time = OFFICE_CONFIG["rounds"] * 5  # ~5 seconds per round
        start_time = time.time()

        while time.time() - start_time < simulation_time:
            # Check leaderboard
            result = client.fetch("leaderboard")
            if result.get("success") and result.get("exists"):
                lb = result["value"]
                if lb and verbose:
                    round_num = lb.get("round", 0)
                    rankings = lb.get("rankings", [])
                    if rankings:
                        print(f"\n[Round {round_num}] Leaderboard:")
                        for i, (name, score, rank) in enumerate(rankings[:3], 1):
                            print(f"  {i}. {name} [{rank}]: {score} pts")

            # Check if simulation ended
            agents = client.list_agents()
            running = [a for a in agents if a.get("state") == "running"]
            if len(running) == 0:
                print("\nAll agents finished!")
                break

            time.sleep(3)

        # Final results
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)

        result = client.fetch("leaderboard")
        if result.get("success") and result.get("exists"):
            lb = result["value"]
            rankings = lb.get("rankings", [])

            print("\nFinal Standings:")
            for i, (name, score, rank) in enumerate(rankings, 1):
                medal = ""
                if i == 1:
                    medal = " [WINNER]"
                elif i == 2:
                    medal = " [2nd]"
                elif i == 3:
                    medal = " [3rd]"

                print(f"  {i}. {name} [{rank}]: {score} pts{medal}")

        # Cleanup
        print("\nCleaning up agents...")
        for agent_name in spawned_agents:
            try:
                client.kill(name=agent_name)
            except:
                pass

        print("\nSimulation complete!")


def main():
    parser = argparse.ArgumentParser(description="Run Office World Simulation")
    parser.add_argument("--workers", type=int, default=5,
                       help="Number of workers (max 8)")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM for worker decisions")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    args = parser.parse_args()

    num_workers = min(args.workers, len(WORKER_NAMES))
    run_simulation(num_workers, args.use_llm, verbose=not args.quiet)


if __name__ == "__main__":
    main()
