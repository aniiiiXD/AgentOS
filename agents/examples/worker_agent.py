#!/usr/bin/env python3
"""
Worker Agent - A simple agent that runs inside a sandbox

This agent:
1. Connects to the kernel
2. Sends periodic heartbeat messages
3. Responds to work requests
"""

import sys
import os
import time

# Add the SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_sdk'))

from agentos import AgentOSClient, SyscallOp


def main():
    socket_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/agentos.sock'

    print(f"[Worker] Starting, connecting to {socket_path}")
    print(f"[Worker] PID: {os.getpid()}")
    print(f"[Worker] Hostname: {os.uname().nodename}")

    client = AgentOSClient(socket_path)

    if not client.connect():
        print("[Worker] ERROR: Failed to connect")
        return 1

    print("[Worker] Connected!")

    # Send a few heartbeat messages
    for i in range(5):
        msg = f"heartbeat #{i+1} from PID {os.getpid()}"
        print(f"[Worker] Sending: {msg}")

        response = client.echo(msg)
        if response:
            print(f"[Worker] Received: {response}")
        else:
            print("[Worker] No response!")

        time.sleep(1)

    # Final think request
    response = client.think("Worker task complete")
    if response:
        print(f"[Worker] Think response: {response}")

    client.disconnect()
    print("[Worker] Done, exiting")

    return 0


if __name__ == '__main__':
    sys.exit(main())
