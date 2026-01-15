#!/usr/bin/env python3
"""
Spawn Test - Test spawning sandboxed agents

This script:
1. Connects to the kernel
2. Spawns a worker agent in a sandbox
3. Lists running agents
4. Waits and kills the agent
"""

import sys
import os
import time

# Add the SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_sdk'))

from agentos import AgentOSClient


def main():
    print("=" * 50)
    print("  AgentOS Sandbox Test")
    print("=" * 50)
    print()

    socket_path = '/tmp/agentos.sock'
    if len(sys.argv) > 1:
        socket_path = sys.argv[1]

    # Path to worker agent
    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, 'worker_agent.py')

    print(f"Socket: {socket_path}")
    print(f"Worker script: {worker_script}")
    print()

    client = AgentOSClient(socket_path)

    if not client.connect():
        print("ERROR: Failed to connect to kernel")
        print("Make sure agentos_kernel is running!")
        return 1

    print("Connected to kernel")
    print()

    # Test 1: List agents (should be empty)
    print("[Test 1] List agents (initial)")
    print("-" * 30)
    agents = client.list_agents()
    print(f"Running agents: {len(agents)}")
    for a in agents:
        print(f"  - {a['name']} (id={a['id']}, pid={a['pid']})")
    print()

    # Test 2: Spawn a worker agent
    print("[Test 2] Spawn worker agent")
    print("-" * 30)
    result = client.spawn(
        name="worker1",
        script=worker_script,
        sandboxed=True,  # Enable sandbox (may need root)
        limits={"memory": 128 * 1024 * 1024}  # 128MB
    )

    if result and 'error' not in result:
        print(f"Spawned: {result['name']} (id={result['id']}, pid={result['pid']})")
        print("PASS!")
    else:
        print(f"Result: {result}")
        if result and 'error' in result:
            print(f"Note: {result['error']}")
            print("(Sandboxing may require root privileges)")
    print()

    # Test 3: List agents after spawn
    print("[Test 3] List agents (after spawn)")
    print("-" * 30)
    agents = client.list_agents()
    print(f"Running agents: {len(agents)}")
    for a in agents:
        print(f"  - {a['name']} (id={a['id']}, pid={a['pid']}, state={a['state']})")
    print()

    # Test 4: Wait for worker to run
    print("[Test 4] Waiting for worker (6 seconds)...")
    print("-" * 30)
    for i in range(6):
        time.sleep(1)
        print(f"  {i+1}s...")

        # Check if still running
        agents = client.list_agents()
        running = [a for a in agents if a.get('running', False)]
        if len(running) == 0 and i > 2:
            print("  Worker finished!")
            break
    print()

    # Test 5: List agents after work
    print("[Test 5] List agents (after work)")
    print("-" * 30)
    agents = client.list_agents()
    print(f"Running agents: {len(agents)}")
    for a in agents:
        print(f"  - {a['name']} (id={a['id']}, pid={a['pid']}, state={a['state']})")
    print()

    # Test 6: Kill agent if still running
    print("[Test 6] Cleanup")
    print("-" * 30)
    if agents:
        for a in agents:
            print(f"Killing agent: {a['name']}")
            killed = client.kill(name=a['name'])
            print(f"  Killed: {killed}")
    else:
        print("No agents to kill")
    print()

    client.disconnect()

    print("=" * 50)
    print("  Test completed!")
    print("=" * 50)

    return 0


if __name__ == '__main__':
    sys.exit(main())
