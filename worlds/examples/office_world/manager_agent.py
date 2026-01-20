#!/usr/bin/env python3
"""
Manager Agent - Office World

The manager assigns tasks, evaluates performance, and decides promotions.
Uses LLM for decision-making when evaluating complex situations.
"""

import sys
import os
import json
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'agents', 'python_sdk'))
from agentos import AgentOS

from office_config import OFFICE_CONFIG, WORKER_PERSONALITIES


class ManagerAgent:
    def __init__(self):
        self.agent = AgentOS("manager")
        self.agent.register_name("manager")

        # Track worker performance
        self.worker_scores = {}
        self.worker_ranks = {}
        self.task_assignments = {}
        self.round_number = 0

        self.agent.write("=" * 60)
        self.agent.write("MANAGER AGENT ONLINE")
        self.agent.write(f"Office: {OFFICE_CONFIG['name']}")
        self.agent.write("=" * 60)

    def initialize_workers(self, worker_names: list):
        """Initialize tracking for all workers."""
        personalities = list(WORKER_PERSONALITIES.keys())

        for i, name in enumerate(worker_names):
            self.worker_scores[name] = 0
            self.worker_ranks[name] = "Junior"
            self.agent.write(f"  Registered worker: {name}")

        # Store initial leaderboard
        self.update_leaderboard()

    def update_leaderboard(self):
        """Update the shared leaderboard in state store."""
        leaderboard = {
            "round": self.round_number,
            "rankings": sorted(
                [(name, score, self.worker_ranks[name])
                 for name, score in self.worker_scores.items()],
                key=lambda x: x[1],
                reverse=True
            ),
            "updated_at": time.strftime("%H:%M:%S")
        }
        self.agent.store("leaderboard", leaderboard, scope="global")

    def generate_tasks(self, num_tasks: int) -> list:
        """Generate random tasks for this round."""
        tasks = []
        task_types = OFFICE_CONFIG["task_types"]

        for i in range(num_tasks):
            task_type = random.choice(task_types)
            task = {
                "id": f"TASK-{self.round_number:02d}-{i:02d}",
                "type": task_type["name"],
                "difficulty": task_type["difficulty"],
                "points": task_type["points"],
                "description": task_type["description"],
                "status": "available",
                "assigned_to": None,
            }
            tasks.append(task)

        return tasks

    def assign_tasks(self, tasks: list):
        """Broadcast available tasks to workers."""
        self.agent.write(f"\n[Round {self.round_number}] Assigning {len(tasks)} tasks...")

        # Store tasks in global state
        self.agent.store("available_tasks", tasks, scope="global")

        # Broadcast to all workers
        self.agent.broadcast({
            "type": "new_tasks",
            "round": self.round_number,
            "tasks": tasks,
            "message": f"New tasks available! Round {self.round_number}"
        })

        for task in tasks:
            self.agent.write(f"  [{task['difficulty'].upper()}] {task['id']}: {task['type']} ({task['points']} pts)")

    def process_task_claims(self, timeout_seconds: float = 5.0):
        """Process task claims from workers."""
        self.agent.write("\nProcessing task claims...")

        start_time = time.time()
        claimed_tasks = {}

        while time.time() - start_time < timeout_seconds:
            messages = self.agent.recv_messages(max_messages=10)

            if messages.get("success") and messages.get("messages"):
                for msg in messages["messages"]:
                    content = msg.get("message", {})

                    if content.get("type") == "claim_task":
                        task_id = content.get("task_id")
                        worker = msg.get("from_name")

                        if task_id not in claimed_tasks:
                            claimed_tasks[task_id] = worker
                            self.task_assignments[task_id] = {
                                "worker": worker,
                                "claimed_at": time.time(),
                                "status": "in_progress"
                            }
                            self.agent.write(f"  {task_id} -> {worker}")

                            # Notify worker of assignment
                            self.agent.send_message({
                                "type": "task_assigned",
                                "task_id": task_id,
                                "message": f"Task {task_id} assigned to you!"
                            }, to_name=worker)
                        else:
                            # Task already claimed
                            self.agent.send_message({
                                "type": "task_taken",
                                "task_id": task_id,
                                "taken_by": claimed_tasks[task_id]
                            }, to_name=worker)

            time.sleep(0.2)

        return claimed_tasks

    def collect_results(self, timeout_seconds: float = 10.0):
        """Collect task completion results from workers."""
        self.agent.write("\nCollecting results...")

        start_time = time.time()
        results = []

        while time.time() - start_time < timeout_seconds:
            messages = self.agent.recv_messages(max_messages=10)

            if messages.get("success") and messages.get("messages"):
                for msg in messages["messages"]:
                    content = msg.get("message", {})

                    if content.get("type") == "task_complete":
                        task_id = content.get("task_id")
                        worker = msg.get("from_name")
                        success = content.get("success", False)
                        quality = content.get("quality", 0.5)

                        result = {
                            "task_id": task_id,
                            "worker": worker,
                            "success": success,
                            "quality": quality,
                        }
                        results.append(result)

                        # Calculate points
                        if success and task_id in self.task_assignments:
                            base_points = content.get("points", 10)
                            actual_points = int(base_points * quality)
                            self.worker_scores[worker] = self.worker_scores.get(worker, 0) + actual_points

                            self.agent.write(f"  {worker} completed {task_id}: +{actual_points} pts (quality: {quality:.0%})")
                        elif not success:
                            self.agent.write(f"  {worker} FAILED {task_id}")

            time.sleep(0.3)

        return results

    def check_promotions(self):
        """Check if any workers qualify for promotion."""
        self.agent.write("\n" + "=" * 40)
        self.agent.write("PROMOTION REVIEW")
        self.agent.write("=" * 40)

        promotions = []
        rules = OFFICE_CONFIG["promotion_rules"]

        for worker, score in self.worker_scores.items():
            current_rank = self.worker_ranks[worker]
            new_rank = current_rank

            if current_rank == "Junior" and score >= rules["junior_to_mid"]:
                new_rank = "Mid-Level"
            elif current_rank == "Mid-Level" and score >= rules["mid_to_senior"]:
                new_rank = "Senior"
            elif current_rank == "Senior" and score >= rules["senior_to_lead"]:
                new_rank = "Team Lead"

            if new_rank != current_rank:
                self.worker_ranks[worker] = new_rank
                promotions.append((worker, current_rank, new_rank))
                self.agent.write(f"  PROMOTED: {worker} ({current_rank} -> {new_rank})")

                # Notify the worker
                self.agent.send_message({
                    "type": "promotion",
                    "old_rank": current_rank,
                    "new_rank": new_rank,
                    "message": f"Congratulations! You've been promoted to {new_rank}!"
                }, to_name=worker)

        if not promotions:
            self.agent.write("  No promotions this review cycle.")

        # Update leaderboard
        self.update_leaderboard()

        return promotions

    def trigger_chaos_event(self):
        """Randomly trigger a chaos event."""
        for event in OFFICE_CONFIG["chaos_events"]:
            if random.random() < event["probability"]:
                self.agent.write(f"\n*** CHAOS EVENT: {event['name'].upper()} ***")
                self.agent.write(f"    Effect: {event['effect']}")

                self.agent.broadcast({
                    "type": "chaos_event",
                    "event": event["name"],
                    "effect": event["effect"]
                })

                return event
        return None

    def run_round(self):
        """Run a single simulation round."""
        self.round_number += 1

        self.agent.write(f"\n{'='*60}")
        self.agent.write(f"ROUND {self.round_number}")
        self.agent.write(f"{'='*60}")

        # Maybe trigger chaos
        self.trigger_chaos_event()

        # Generate and assign tasks
        tasks = self.generate_tasks(OFFICE_CONFIG["tasks_per_round"])
        self.assign_tasks(tasks)

        # Wait for claims
        time.sleep(1)
        self.process_task_claims(timeout_seconds=3.0)

        # Wait for work to complete
        time.sleep(2)
        self.collect_results(timeout_seconds=5.0)

        # Check promotions periodically
        if self.round_number % OFFICE_CONFIG["promotion_check_interval"] == 0:
            self.check_promotions()

        # Show current standings
        self.show_standings()

    def show_standings(self):
        """Display current standings."""
        self.agent.write("\nCurrent Standings:")
        sorted_workers = sorted(self.worker_scores.items(), key=lambda x: x[1], reverse=True)

        for i, (worker, score) in enumerate(sorted_workers, 1):
            rank = self.worker_ranks[worker]
            self.agent.write(f"  {i}. {worker} [{rank}]: {score} pts")

    def end_simulation(self):
        """End the simulation and announce final results."""
        self.agent.write("\n" + "=" * 60)
        self.agent.write("SIMULATION COMPLETE")
        self.agent.write("=" * 60)

        # Final promotion check
        self.check_promotions()

        # Announce winner
        sorted_workers = sorted(self.worker_scores.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_workers[0]

        self.agent.write(f"\nWINNER: {winner[0]} with {winner[1]} points!")
        self.agent.write(f"Final Rank: {self.worker_ranks[winner[0]]}")

        # Broadcast end
        self.agent.broadcast({
            "type": "simulation_end",
            "winner": winner[0],
            "final_scores": dict(self.worker_scores),
            "final_ranks": dict(self.worker_ranks)
        })

        self.agent.write("\nFinal Rankings:")
        for i, (worker, score) in enumerate(sorted_workers, 1):
            rank = self.worker_ranks[worker]
            self.agent.write(f"  {i}. {worker} [{rank}]: {score} pts")


def main():
    manager = ManagerAgent()

    # Wait for workers to register
    manager.agent.write("\nWaiting for workers to join...")
    time.sleep(3)

    # Get registered workers from state
    result = manager.agent.fetch("registered_workers")
    if result.get("success") and result.get("exists"):
        workers = result["value"]
        manager.initialize_workers(workers)
    else:
        # Default workers if none registered
        manager.initialize_workers(["alice", "bob", "charlie", "diana", "eve"])

    # Run simulation
    for _ in range(OFFICE_CONFIG["rounds"]):
        manager.run_round()
        time.sleep(1)

    manager.end_simulation()
    manager.agent.exit(0)


if __name__ == "__main__":
    main()
