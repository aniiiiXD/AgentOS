#!/usr/bin/env python3
"""
Office World Configuration

Defines the virtual office environment where agents compete for promotion.
"""

OFFICE_CONFIG = {
    "name": "TechCorp Office",
    "description": "A competitive office where workers vie for promotion",

    # Virtual filesystem for the office
    "filesystem": {
        "/office/tasks": "Task assignment directory",
        "/office/reports": "Completed work reports",
        "/office/announcements": "Manager announcements",
        "/office/leaderboard": "Performance rankings",
        "/office/inbox": "Inter-office messages",
    },

    # Task types workers can receive
    "task_types": [
        {
            "name": "code_review",
            "difficulty": "easy",
            "points": 10,
            "description": "Review a pull request and provide feedback"
        },
        {
            "name": "bug_fix",
            "difficulty": "medium",
            "points": 25,
            "description": "Fix a reported bug in the system"
        },
        {
            "name": "feature_implementation",
            "difficulty": "hard",
            "points": 50,
            "description": "Implement a new feature from spec"
        },
        {
            "name": "documentation",
            "difficulty": "easy",
            "points": 15,
            "description": "Write documentation for a module"
        },
        {
            "name": "optimization",
            "difficulty": "hard",
            "points": 40,
            "description": "Optimize performance of a slow component"
        },
        {
            "name": "client_meeting",
            "difficulty": "medium",
            "points": 30,
            "description": "Attend and summarize a client meeting"
        },
    ],

    # Promotion thresholds
    "promotion_rules": {
        "junior_to_mid": 100,      # Points needed
        "mid_to_senior": 300,
        "senior_to_lead": 600,
    },

    # Worker starting ranks
    "ranks": ["Junior", "Mid-Level", "Senior", "Team Lead"],

    # Chaos events that can occur
    "chaos_events": [
        {"name": "server_outage", "probability": 0.1, "effect": "All tasks paused for 1 round"},
        {"name": "urgent_deadline", "probability": 0.15, "effect": "Double points but must complete in 1 round"},
        {"name": "team_lunch", "probability": 0.05, "effect": "All workers get +5 morale bonus"},
        {"name": "coffee_machine_broken", "probability": 0.1, "effect": "Productivity -20% this round"},
    ],

    # Simulation parameters
    "rounds": 10,
    "tasks_per_round": 3,
    "promotion_check_interval": 3,  # Check for promotions every N rounds
}

# Worker personalities (affects decision making)
WORKER_PERSONALITIES = {
    "ambitious": {
        "risk_tolerance": 0.8,      # Takes harder tasks
        "collaboration": 0.3,       # Less likely to help others
        "work_ethic": 0.9,
    },
    "team_player": {
        "risk_tolerance": 0.5,
        "collaboration": 0.9,       # Helps teammates
        "work_ethic": 0.7,
    },
    "steady": {
        "risk_tolerance": 0.3,      # Prefers easy tasks
        "collaboration": 0.5,
        "work_ethic": 0.8,
    },
    "perfectionist": {
        "risk_tolerance": 0.6,
        "collaboration": 0.4,
        "work_ethic": 1.0,          # Always completes tasks
    },
    "slacker": {
        "risk_tolerance": 0.2,
        "collaboration": 0.6,
        "work_ethic": 0.4,          # Sometimes fails tasks
    },
}
