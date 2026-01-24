#!/usr/bin/env python3
"""
Test script for hot reload / auto-recovery functionality.

Tests:
1. restart_policy="on-failure" restarts on non-zero exit
2. restart_policy="always" restarts on any exit
3. restart_policy="never" does not restart
4. max_restarts limit triggers escalation
5. Exponential backoff timing
"""
import sys
import os
import time

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents', 'python_sdk'))

from agentos import CloveClient

CRASH_SCRIPT = os.path.join(os.path.dirname(__file__), 'crash_agent.py')
SUCCESS_SCRIPT = os.path.join(os.path.dirname(__file__), 'success_agent.py')


def test_on_failure_policy():
    """Test that on-failure policy restarts crashed agents."""
    print("\n" + "=" * 60)
    print("TEST 1: restart_policy='on-failure' with crashing agent")
    print("=" * 60)

    with CloveClient() as client:
        # Subscribe to restart events
        result = client.subscribe(["AGENT_RESTARTING", "AGENT_ESCALATED", "AGENT_SPAWNED"])
        print(f"Subscribed to events: {result}")

        # Spawn agent with on-failure restart policy
        result = client.spawn(
            name="crasher",
            script=CRASH_SCRIPT,
            sandboxed=False,
            restart_policy="on-failure",
            max_restarts=3,
            restart_window=60
        )
        print(f"Spawned crasher: {result}")

        if not result or "error" in result:
            print("FAILED: Could not spawn agent")
            return False

        # Wait and watch for restarts
        restart_count = 0
        escalated = False

        for i in range(45):  # 45 seconds max
            time.sleep(1)

            # Check events
            events = client.poll_events(max_events=10)
            if events.get("count", 0) > 0:
                for event in events.get("events", []):
                    print(f"  Event: {event['type']} - {event.get('data', {})}")
                    if event["type"] == "AGENT_RESTARTING":
                        restart_count += 1
                    elif event["type"] == "AGENT_ESCALATED":
                        escalated = True

            # Check agent status
            agents = client.list_agents()
            agent_info = [a for a in agents if a.get("name") == "crasher"]
            if agent_info:
                print(f"  [{i}s] Agent status: running, restarts so far: {restart_count}")
            else:
                print(f"  [{i}s] Agent not running, restarts so far: {restart_count}")

            if escalated:
                print("  Agent escalated (max restarts reached)")
                break

        # Clean up
        client.kill(name="crasher")

        # Verify results
        if restart_count >= 2 and escalated:
            print("PASSED: Agent was restarted and eventually escalated")
            return True
        else:
            print(f"FAILED: Expected restarts and escalation, got restarts={restart_count}, escalated={escalated}")
            return False


def test_never_policy():
    """Test that never policy does not restart."""
    print("\n" + "=" * 60)
    print("TEST 2: restart_policy='never' should not restart")
    print("=" * 60)

    with CloveClient() as client:
        result = client.spawn(
            name="no_restart",
            script=CRASH_SCRIPT,
            sandboxed=False,
            restart_policy="never"
        )
        print(f"Spawned no_restart: {result}")

        if not result or "error" in result:
            print("FAILED: Could not spawn agent")
            return False

        # Wait for agent to crash
        time.sleep(4)

        # Check if agent is gone and not restarted
        agents = client.list_agents()
        agent_info = [a for a in agents if a.get("name") == "no_restart"]

        if not agent_info:
            print("PASSED: Agent exited and was not restarted")
            return True
        else:
            print("FAILED: Agent was unexpectedly restarted")
            client.kill(name="no_restart")
            return False


def test_always_policy():
    """Test that always policy restarts even on success."""
    print("\n" + "=" * 60)
    print("TEST 3: restart_policy='always' restarts on exit code 0")
    print("=" * 60)

    with CloveClient() as client:
        result = client.subscribe(["AGENT_RESTARTING"])

        result = client.spawn(
            name="always_restart",
            script=SUCCESS_SCRIPT,
            sandboxed=False,
            restart_policy="always",
            max_restarts=2
        )
        print(f"Spawned always_restart: {result}")

        if not result or "error" in result:
            print("FAILED: Could not spawn agent")
            return False

        restart_count = 0
        for i in range(15):
            time.sleep(1)

            events = client.poll_events()
            for event in events.get("events", []):
                if event["type"] == "AGENT_RESTARTING":
                    restart_count += 1
                    print(f"  Restart #{restart_count} detected")

            if restart_count >= 1:
                break

        client.kill(name="always_restart")

        if restart_count >= 1:
            print("PASSED: Agent with exit code 0 was restarted")
            return True
        else:
            print("FAILED: Agent was not restarted despite 'always' policy")
            return False


def test_spawn_with_restart_response():
    """Test that spawn response includes restart_policy."""
    print("\n" + "=" * 60)
    print("TEST 4: spawn() response includes restart_policy")
    print("=" * 60)

    with CloveClient() as client:
        result = client.spawn(
            name="policy_check",
            script=SUCCESS_SCRIPT,
            sandboxed=False,
            restart_policy="on-failure"
        )
        print(f"Spawn result: {result}")

        client.kill(name="policy_check")

        if result and result.get("restart_policy") == "on-failure":
            print("PASSED: restart_policy included in response")
            return True
        else:
            print("FAILED: restart_policy not in response or incorrect")
            return False


def main():
    print("=" * 60)
    print("HOT RELOAD / AUTO-RECOVERY TEST SUITE")
    print("=" * 60)
    print(f"Crash script: {CRASH_SCRIPT}")
    print(f"Success script: {SUCCESS_SCRIPT}")

    results = []

    # Run tests
    results.append(("spawn response includes restart_policy", test_spawn_with_restart_response()))
    results.append(("never policy", test_never_policy()))
    results.append(("always policy", test_always_policy()))
    results.append(("on-failure policy with escalation", test_on_failure_policy()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {status}: {name}")
        if result:
            passed += 1

    print(f"\n{passed}/{len(results)} tests passed")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
