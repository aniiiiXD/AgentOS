#!/usr/bin/env python3
"""Test agent that exits successfully after a short delay."""
import time
import sys

print("Success agent starting...")
time.sleep(2)
print("Success agent exiting with code 0!")
sys.exit(0)
