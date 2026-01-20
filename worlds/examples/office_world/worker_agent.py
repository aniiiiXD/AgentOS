#!/usr/bin/env python3
"""
Worker Agent - Office World

A worker who competes for promotion by completing tasks.
Each worker has a personality that affects their behavior.
Can optionally use LLM for task completion decisions.
"""

import sys
import os
import json
import random
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'agents', 'python_sdk'))
from agentos import AgentOS

from office_config import WORKER_PERSONALITIES


class WorkerAgent:
    def __init__(self, name: str, personality: str = "steady", use_llm: bool = False):
        self.name = name
        self.personality = WORKER_PERSONALITIES.get(personality, WORKER_PERSONALITIES["steady"])
        self.personality_name = personality
        self.use_llm = use_llm

        self.agent = AgentOS(name)
        self.agent.register_name(name)

        self.score = 0
        self.rank = "Junior"
        self.current_task = None
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.running = True

        self.agent.write(f"[{self.name}] Worker online! Personality: {personality}")
        self.agent.write(f"  Risk tolerance: {self.personality['risk_tolerance']:.0%}")
        self.agent.write(f"  Collaboration: {self.personality['collaboration']:.0%}")
        self.agent.write(f"  Work ethic: {self.personality['work_ethic']:.0%}")

    def register_with_manager(self):
        """Register this worker in the global worker list."""
        # Get current worker list
        result = self.agent.fetch("registered_workers")
        if result.get("success") and result.get("exists"):
            workers = result["value"]
        else:
            workers = []

        # Add self if not already registered
        if self.name not in workers:
            workers.append(self.name)
            self.agent.store("registered_workers", workers, scope="global")
            self.agent.write(f"[{self.name}] Registered with office")

    def decide_task_claim(self, tasks: list) -> dict:
        """Decide which task to claim based on personality."""
        if not tasks:
            return None

        available = [t for t in tasks if t.get("status") == "available"]
        if not available:
            return None

        # Score each task based on personality
        scored_tasks = []
        for task in available:
            score = 0
            difficulty = task.get("difficulty", "medium")
            points = task.get("points", 10)

            # Risk tolerance affects preference for harder tasks
            if difficulty == "hard":
                score = points * self.personality["risk_tolerance"]
            elif difficulty == "medium":
                score = points * 0.7
            else:  # easy
                score = points * (1 - self.personality["risk_tolerance"] * 0.5)

            scored_tasks.append((task, score))

        # Sort by score and pick best (with some randomness)
        scored_tasks.sort(key=lambda x: x[1], reverse=True)

        # Top performer might pick best, others have variance
        if random.random() < self.personality["work_ethic"]:
            return scored_tasks[0][0]
        else:
            # Pick from top 3
            top_tasks = scored_tasks[:min(3, len(scored_tasks))]
            return random.choice(top_tasks)[0]

    def work_on_task(self, task: dict) -> tuple:
        """
        Simulate working on a task.
        Returns (success, quality) based on personality and task difficulty.
        """
        difficulty = task.get("difficulty", "medium")
        base_success_rate = {
            "easy": 0.95,
            "medium": 0.75,
            "hard": 0.55,
        }.get(difficulty, 0.75)

        # Work ethic improves success rate
        success_rate = base_success_rate + (self.personality["work_ethic"] * 0.2)
        success_rate = min(0.98, success_rate)  # Cap at 98%

        # Simulate work time
        work_time = random.uniform(0.5, 2.0)

        if self.use_llm:
            # Use LLM to "think" about the task
            prompt = f"""You are {self.name}, a {self.personality_name} worker.
            Task: {task['description']}
            Difficulty: {difficulty}

            Briefly describe (1-2 sentences) how you would approach this task."""

            result = self.agent.think(prompt)
            if result.get("success"):
                self.agent.write(f"[{self.name}] Thinking: {result.get('content', '')[:100]}...")

        time.sleep(work_time)

        # Determine success
        success = random.random() < success_rate

        # Quality varies based on work ethic
        if success:
            base_quality = random.uniform(0.6, 1.0)
            quality = base_quality * (0.5 + self.personality["work_ethic"] * 0.5)
            quality = min(1.0, quality)
        else:
            quality = 0

        return success, quality

    def handle_message(self, msg: dict):
        """Handle incoming message from manager or other workers."""
        content = msg.get("message", {})
        msg_type = content.get("type")
        sender = msg.get("from_name", "unknown")

        if msg_type == "new_tasks":
            # New tasks available - decide what to claim
            tasks = content.get("tasks", [])
            self.agent.write(f"[{self.name}] Received {len(tasks)} new tasks")

            chosen_task = self.decide_task_claim(tasks)
            if chosen_task:
                self.agent.write(f"[{self.name}] Claiming: {chosen_task['id']} ({chosen_task['type']})")
                self.agent.send_message({
                    "type": "claim_task",
                    "task_id": chosen_task["id"]
                }, to_name="manager")
                self.current_task = chosen_task

        elif msg_type == "task_assigned":
            # Task was assigned to us
            task_id = content.get("task_id")
            self.agent.write(f"[{self.name}] Got assignment: {task_id}")

            # Work on task
            if self.current_task and self.current_task["id"] == task_id:
                success, quality = self.work_on_task(self.current_task)

                # Report completion
                self.agent.send_message({
                    "type": "task_complete",
                    "task_id": task_id,
                    "success": success,
                    "quality": quality,
                    "points": self.current_task.get("points", 10)
                }, to_name="manager")

                if success:
                    self.tasks_completed += 1
                    self.agent.write(f"[{self.name}] Completed {task_id} (quality: {quality:.0%})")
                else:
                    self.tasks_failed += 1
                    self.agent.write(f"[{self.name}] FAILED {task_id}")

                self.current_task = None

        elif msg_type == "task_taken":
            # Someone else got the task we wanted
            taken_by = content.get("taken_by")
            task_id = content.get("task_id")
            self.agent.write(f"[{self.name}] Task {task_id} taken by {taken_by}")
            self.current_task = None

        elif msg_type == "promotion":
            # We got promoted!
            old_rank = content.get("old_rank")
            new_rank = content.get("new_rank")
            self.rank = new_rank
            self.agent.write(f"[{self.name}] *** PROMOTED: {old_rank} -> {new_rank}! ***")

        elif msg_type == "chaos_event":
            # Chaos event occurred
            event = content.get("event")
            self.agent.write(f"[{self.name}] Chaos! {event}")

        elif msg_type == "simulation_end":
            # Simulation is over
            winner = content.get("winner")
            final_scores = content.get("final_scores", {})
            my_score = final_scores.get(self.name, 0)

            self.agent.write(f"[{self.name}] Simulation ended. My score: {my_score}")
            if winner == self.name:
                self.agent.write(f"[{self.name}] *** I WON! ***")

            self.running = False

        elif msg_type == "help_request":
            # Another worker asking for help
            if random.random() < self.personality["collaboration"]:
                self.agent.write(f"[{self.name}] Helping {sender}")
                self.agent.send_message({
                    "type": "help_response",
                    "helping": True
                }, to_name=sender)

    def check_leaderboard(self):
        """Check current standings."""
        result = self.agent.fetch("leaderboard")
        if result.get("success") and result.get("exists"):
            leaderboard = result["value"]
            rankings = leaderboard.get("rankings", [])

            # Find our position
            for i, (name, score, rank) in enumerate(rankings, 1):
                if name == self.name:
                    self.score = score
                    self.rank = rank
                    return i

        return -1

    def run(self):
        """Main worker loop."""
        self.register_with_manager()

        while self.running:
            # Check for messages
            messages = self.agent.recv_messages(max_messages=5)

            if messages.get("success") and messages.get("messages"):
                for msg in messages["messages"]:
                    self.handle_message(msg)

            # Occasionally check leaderboard
            if random.random() < 0.1:
                position = self.check_leaderboard()
                if position > 0:
                    self.agent.write(f"[{self.name}] Leaderboard position: #{position} ({self.score} pts)")

            time.sleep(0.3)

        # Final stats
        self.agent.write(f"[{self.name}] Final stats: {self.tasks_completed} completed, {self.tasks_failed} failed")
        self.agent.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Office Worker Agent")
    parser.add_argument("--name", required=True, help="Worker name")
    parser.add_argument("--personality", default="steady",
                       choices=list(WORKER_PERSONALITIES.keys()),
                       help="Worker personality type")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM for task decisions")
    args = parser.parse_args()

    worker = WorkerAgent(args.name, args.personality, args.use_llm)
    worker.run()


if __name__ == "__main__":
    main()
