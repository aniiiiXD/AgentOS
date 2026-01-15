#!/usr/bin/env python3
"""
Hello Agent - Basic AgentOS test agent

This agent connects to the AgentOS kernel and tests basic IPC functionality.
"""

import sys
import os

# Add the SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_sdk'))

from agentos import AgentOSClient, SyscallOp


def main():
    print("=" * 50)
    print("  Hello Agent - AgentOS Test Client")
    print("=" * 50)
    print()

    socket_path = '/tmp/agentos.sock'
    if len(sys.argv) > 1:
        socket_path = sys.argv[1]

    print(f"Connecting to: {socket_path}")

    client = AgentOSClient(socket_path)

    if not client.connect():
        print("ERROR: Failed to connect to kernel")
        print("Make sure agentos_kernel is running!")
        return 1

    print("Connected!")
    print()

    # Test 1: Echo (NOOP)
    print("[Test 1] Echo (SYS_NOOP)")
    print("-" * 30)
    test_message = "Hello from Python agent!"
    print(f"Sending: {test_message}")

    response = client.echo(test_message)
    if response:
        print(f"Received: {response}")
        if response == test_message:
            print("PASS: Echo successful!")
        else:
            print("WARN: Response differs from sent message")
    else:
        print("FAIL: No response")
    print()

    # Test 2: Multiple messages
    print("[Test 2] Multiple Messages")
    print("-" * 30)
    messages = ["Message 1", "Message 2", "Message 3"]
    for i, msg in enumerate(messages):
        response = client.echo(msg)
        status = "OK" if response == msg else "FAIL"
        print(f"  {i+1}. '{msg}' -> '{response}' [{status}]")
    print()

    # Test 3: Think (placeholder for LLM)
    print("[Test 3] Think (SYS_THINK)")
    print("-" * 30)
    prompt = "What is the meaning of life?"
    print(f"Sending: {prompt}")

    response = client.think(prompt)
    if response:
        print(f"Received: {response}")
        print("PASS: Think call successful!")
    else:
        print("FAIL: No response")
    print()

    # Test 4: Large message
    print("[Test 4] Large Message (1KB)")
    print("-" * 30)
    large_message = "X" * 1024
    response = client.echo(large_message)
    if response and len(response) == 1024:
        print(f"Sent/Received: {len(large_message)} bytes")
        print("PASS: Large message successful!")
    else:
        print("FAIL: Large message failed")
    print()

    # Test 5: Exit
    print("[Test 5] Exit (SYS_EXIT)")
    print("-" * 30)
    print("Sending exit request...")
    response = client.call(SyscallOp.SYS_EXIT)
    if response:
        print(f"Received: {response.payload_str}")
        print("PASS: Exit acknowledged!")
    else:
        print("FAIL: No response")
    print()

    client.disconnect()
    print("=" * 50)
    print("  All tests completed!")
    print("=" * 50)

    return 0


if __name__ == '__main__':
    sys.exit(main())
