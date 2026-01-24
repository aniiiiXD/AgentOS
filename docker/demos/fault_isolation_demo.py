#!/usr/bin/env python3
"""
Fault Isolation Demo - Docker version

Spawns three agents with different behavior and demonstrates
OS-level isolation via cgroups and namespaces.
"""

import sys
import os
import time

from clove_sdk import CloveClient

DEMO_DURATION = 30
SCRIPT_DIR = "/app/agents"

AGENTS = {
    "cpu-hog": {
        "script": os.path.join(SCRIPT_DIR, "cpu_hog_agent.py"),
        "limits": {
            "memory": 64 * 1024 * 1024,
            "cpu_quota": 10000,
            "max_pids": 4
        },
        "description": "CPU burner (should be throttled to 10%)"
    },
    "mem-hog": {
        "script": os.path.join(SCRIPT_DIR, "memory_hog_agent.py"),
        "limits": {
            "memory": 50 * 1024 * 1024,
            "cpu_quota": 100000,
            "max_pids": 4
        },
        "description": "Memory leaker (should be OOM-killed at 50MB)"
    },
    "healthy": {
        "script": os.path.join(SCRIPT_DIR, "healthy_agent.py"),
        "limits": {
            "memory": 128 * 1024 * 1024,
            "cpu_quota": 100000,
            "max_pids": 16
        },
        "description": "Well-behaved agent (should survive)"
    }
}


def main():
    print("=" * 60)
    print("  Clove Fault Isolation Demo")
    print("=" * 60)
    print()

    client = CloveClient()
    if not client.connect():
        print("ERROR: Failed to connect to kernel")
        return 1

    print("Connected to kernel.\n")

    agents_status = {}

    print("Phase 1: Spawning Agents")
    print("-" * 40)

    for name, config in AGENTS.items():
        print(f"  Spawning {name}: {config['description']}")
        result = client.spawn(
            name=name,
            script=config['script'],
            sandboxed=True,
            network=False,
            limits=config['limits']
        )

        if result and result.get('status') == 'running':
            agents_status[name] = {'status': 'RUNNING', 'pid': result.get('pid')}
            print(f"    -> PID={result.get('pid')}")
        else:
            agents_status[name] = {'status': 'FAILED'}
            print(f"    -> Failed: {result}")
        time.sleep(0.5)

    print(f"\nPhase 2: Monitoring ({DEMO_DURATION}s)")
    print("-" * 40)

    start_time = time.time()
    try:
        while time.time() - start_time < DEMO_DURATION:
            elapsed = time.time() - start_time
            current_agents = client.list_agents()
            running_names = set()
            if isinstance(current_agents, list):
                for agent in current_agents:
                    running_names.add(agent.get('name'))

            for name in AGENTS.keys():
                was_running = agents_status[name]['status'] == 'RUNNING'
                is_running = name in running_names

                if was_running and not is_running:
                    if name == 'mem-hog':
                        agents_status[name]['status'] = 'KILLED'
                        print(f"  [{elapsed:5.1f}s] mem-hog KILLED (OOM)")
                    elif name == 'cpu-hog':
                        agents_status[name]['status'] = 'KILLED'
                        print(f"  [{elapsed:5.1f}s] cpu-hog KILLED")
                    else:
                        agents_status[name]['status'] = 'STOPPED'
                        print(f"  [{elapsed:5.1f}s] {name} stopped")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    # Cleanup
    for name in AGENTS.keys():
        if agents_status[name]['status'] == 'RUNNING':
            client.kill(name=name)

    print(f"\nResults")
    print("-" * 40)

    healthy_ok = agents_status['healthy']['status'] in ['RUNNING', 'STOPPED']
    mem_killed = agents_status['mem-hog']['status'] == 'KILLED'

    print(f"  healthy agent:  {'SURVIVED' if healthy_ok else 'FAILED'}")
    print(f"  mem-hog agent:  {'OOM-KILLED' if mem_killed else agents_status['mem-hog']['status']}")
    print(f"  cpu-hog agent:  {agents_status['cpu-hog']['status']}")
    print()

    if healthy_ok:
        print("  Fault isolation works: healthy agent survived misbehaving neighbors.")
    else:
        print("  Note: Run with --privileged for full cgroups isolation.")

    client.disconnect()
    return 0


if __name__ == '__main__':
    sys.exit(main())
