#!/usr/bin/env python3
"""Test agent that crashes after a short delay."""
import time
import sys

print("Crash agent starting...")
print(f"PID: {sys.argv[0] if len(sys.argv) > 0 else 'unknown'}")
time.sleep(2)
print("Crash agent exiting with code 1!")
sys.exit(1)
