#!/usr/bin/env python3
"""
Systems World Configuration

Defines the virtual infrastructure environment for SRE simulation.
"""

SYSTEMS_CONFIG = {
    "name": "CloudInfra Production",
    "description": "A simulated production infrastructure with servers, services, and chaos",

    # Virtual servers in the infrastructure
    "servers": [
        {
            "name": "web-1",
            "type": "web",
            "cpu_capacity": 100,
            "memory_capacity": 8192,  # MB
            "services": ["nginx", "app-frontend"],
            "health": 100,
        },
        {
            "name": "web-2",
            "type": "web",
            "cpu_capacity": 100,
            "memory_capacity": 8192,
            "services": ["nginx", "app-frontend"],
            "health": 100,
        },
        {
            "name": "api-1",
            "type": "api",
            "cpu_capacity": 100,
            "memory_capacity": 16384,
            "services": ["app-api", "redis-client"],
            "health": 100,
        },
        {
            "name": "db-primary",
            "type": "database",
            "cpu_capacity": 100,
            "memory_capacity": 32768,
            "services": ["postgresql"],
            "health": 100,
            "is_primary": True,
        },
        {
            "name": "db-replica",
            "type": "database",
            "cpu_capacity": 100,
            "memory_capacity": 32768,
            "services": ["postgresql"],
            "health": 100,
            "is_primary": False,
        },
        {
            "name": "cache-1",
            "type": "cache",
            "cpu_capacity": 50,
            "memory_capacity": 16384,
            "services": ["redis"],
            "health": 100,
        },
    ],

    # Services and their dependencies
    "service_dependencies": {
        "app-frontend": ["app-api"],
        "app-api": ["postgresql", "redis"],
        "postgresql": [],
        "redis": [],
        "nginx": ["app-frontend"],
    },

    # Metrics to track
    "metrics": [
        "cpu_usage",
        "memory_usage",
        "request_rate",
        "error_rate",
        "latency_p50",
        "latency_p99",
        "connections",
        "disk_usage",
    ],

    # Alert thresholds
    "alert_thresholds": {
        "cpu_usage": {"warning": 70, "critical": 90},
        "memory_usage": {"warning": 75, "critical": 95},
        "error_rate": {"warning": 1, "critical": 5},
        "latency_p99": {"warning": 500, "critical": 2000},
        "disk_usage": {"warning": 80, "critical": 95},
    },

    # Chaos scenarios that can occur
    "chaos_scenarios": [
        {
            "name": "cpu_spike",
            "description": "Sudden CPU spike on a server",
            "probability": 0.15,
            "duration": 3,  # rounds
            "effect": {"cpu_usage": 95},
        },
        {
            "name": "memory_leak",
            "description": "Memory leak causing gradual increase",
            "probability": 0.1,
            "duration": 5,
            "effect": {"memory_growth": 10},  # +10% per round
        },
        {
            "name": "network_partition",
            "description": "Network partition isolating a server",
            "probability": 0.05,
            "duration": 2,
            "effect": {"isolated": True},
        },
        {
            "name": "disk_full",
            "description": "Disk filling up",
            "probability": 0.08,
            "duration": 4,
            "effect": {"disk_usage": 98},
        },
        {
            "name": "service_crash",
            "description": "A service crashes unexpectedly",
            "probability": 0.12,
            "duration": 1,
            "effect": {"service_down": True},
        },
        {
            "name": "traffic_surge",
            "description": "10x traffic spike",
            "probability": 0.1,
            "duration": 3,
            "effect": {"traffic_multiplier": 10},
        },
        {
            "name": "database_slowdown",
            "description": "Database queries running slow",
            "probability": 0.1,
            "duration": 4,
            "effect": {"db_latency_multiplier": 5},
        },
    ],

    # Remediation actions SRE can take
    "remediation_actions": [
        {
            "name": "restart_service",
            "description": "Restart a crashed or stuck service",
            "success_rate": 0.9,
            "cooldown": 2,
        },
        {
            "name": "scale_up",
            "description": "Add more instances of a service",
            "success_rate": 0.95,
            "cooldown": 3,
        },
        {
            "name": "failover",
            "description": "Failover to replica/backup",
            "success_rate": 0.85,
            "cooldown": 1,
        },
        {
            "name": "clear_cache",
            "description": "Clear cache to free memory",
            "success_rate": 0.98,
            "cooldown": 1,
        },
        {
            "name": "rate_limit",
            "description": "Enable rate limiting",
            "success_rate": 0.95,
            "cooldown": 1,
        },
        {
            "name": "rollback",
            "description": "Rollback to previous version",
            "success_rate": 0.8,
            "cooldown": 5,
        },
    ],

    # Simulation parameters
    "rounds": 15,
    "base_traffic": 1000,  # requests per round
    "sla_target": 99.9,  # availability percentage
}

# SRE Agent skill levels
SRE_SKILL_LEVELS = {
    "junior": {
        "diagnosis_accuracy": 0.6,
        "remediation_speed": 0.5,
        "panic_threshold": 2,  # alerts before overwhelmed
    },
    "mid": {
        "diagnosis_accuracy": 0.8,
        "remediation_speed": 0.7,
        "panic_threshold": 4,
    },
    "senior": {
        "diagnosis_accuracy": 0.95,
        "remediation_speed": 0.9,
        "panic_threshold": 8,
    },
}
