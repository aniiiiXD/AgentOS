#!/usr/bin/env python3
"""
Thinking Agent Example - AgentOS Phase 3

Demonstrates using the SYS_THINK syscall to interact with Gemini LLM.
Requires GEMINI_API_KEY environment variable to be set.
"""

import sys
import os

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_sdk'))

from agentos import AgentOSClient


def main():
    print("=" * 60)
    print("AgentOS Thinking Agent - Gemini LLM Integration")
    print("=" * 60)
    print()

    # Connect to kernel
    with AgentOSClient() as client:
        print("[Connected to AgentOS kernel]")
        print()

        # Test prompts
        prompts = [
            "What is 2 + 2? Reply with just the number.",
            "Write a haiku about computers.",
            "Explain recursion in one sentence.",
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"[Test {i}] Prompt: {prompt}")
            print("-" * 40)

            result = client.think(prompt)

            if result.get("error"):
                print(f"ERROR: {result['error']}")
            else:
                print(f"Response: {result.get('content', '(empty)')}")
                if result.get("tokens"):
                    print(f"Tokens: {result['tokens']}")

            print()

        # Interactive mode
        print("=" * 60)
        print("Interactive Mode (type 'exit' to quit)")
        print("=" * 60)
        print()

        while True:
            try:
                prompt = input("You: ").strip()
                if not prompt:
                    continue
                if prompt.lower() == 'exit':
                    print("Goodbye!")
                    break

                result = client.think(prompt)

                if result.get("error"):
                    print(f"Error: {result['error']}")
                else:
                    print(f"AI: {result.get('content', '(no response)')}")
                    if result.get("tokens"):
                        print(f"   [{result['tokens']} tokens]")

                print()

            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break


if __name__ == '__main__':
    main()
