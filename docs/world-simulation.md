# World Simulation

Clove's sandbox primitives can create **controlled virtual worlds** for agent testing. Worlds provide filesystem virtualization, network mocking, and chaos event injection.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      World Definition                           │
│  {                                                              │
│    "name": "e-commerce-sandbox",                                │
│    "filesystem": {                                              │
│      "/app": "simulated web application",                       │
│      "/db": "mock database with test data",                     │
│      "/logs": "writable log directory"                          │
│    },                                                           │
│    "network": {                                                 │
│      "allowed_hosts": ["api.stripe.test", "db.local"],          │
│      "latency_ms": 50,                                         │
│      "failure_rate": 0.01                                       │
│    },                                                           │
│    "events": [                                                  │
│      {"at": "+1h", "type": "db_failure", "duration": "5m"},    │
│      {"at": "+2h", "type": "traffic_spike", "multiplier": 10}  │
│    ]                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

```python
from clove_sdk import CloveClient

with CloveClient() as client:
    # Create a world
    world = client.world_create("test-world", config={
        "virtual_filesystem": {
            "initial_files": {"/app/config.json": '{"debug": true}'},
            "writable_patterns": ["/tmp/*", "/app/logs/*"]
        },
        "network": {
            "mode": "mock",
            "mock_responses": {
                "https://api.example.com/health": {"status": "ok"}
            }
        },
        "chaos": {
            "enabled": True,
            "failure_rate": 0.05
        }
    })

    world_id = world["world_id"]

    # Join the world
    client.world_join(world_id)

    # Inject chaos events
    client.world_event(world_id, "disk_full")
    client.world_event(world_id, "network_partition")
    client.world_event(world_id, "slow_io", {"latency_ms": 500})

    # Check world state
    state = client.world_state(world_id)
    print(state)

    # Snapshot and restore
    snapshot = client.world_snapshot(world_id)
    client.world_restore(snapshot)

    # Leave and destroy
    client.world_leave()
    client.world_destroy(world_id)
```

## Chaos Events

| Event Type | Description | Parameters |
|-----------|-------------|------------|
| `disk_full` | Simulate full disk | — |
| `disk_fail` | Simulate disk I/O failure | — |
| `network_partition` | Simulate network outage | — |
| `slow_io` | Inject I/O latency | `latency_ms` |

## Capabilities

- **Filesystem virtualization** — mount custom directory trees, inject files, track modifications
- **Network simulation** — mock APIs, inject latency/failures, restrict connectivity
- **Event injection** — trigger failures, load spikes, data corruption at specific times
- **State snapshots** — save/restore world state for reproducible testing

## Use Cases

- Test SRE agents against simulated outages
- Evaluate coding agents in realistic project environments
- Benchmark multi-agent collaboration in shared worlds
- Train agents on edge cases without risking production systems

## Syscalls

| Opcode | Name | Description |
|--------|------|-------------|
| 0xA0 | SYS_WORLD_CREATE | Create world from config |
| 0xA1 | SYS_WORLD_DESTROY | Destroy world |
| 0xA2 | SYS_WORLD_LIST | List active worlds |
| 0xA3 | SYS_WORLD_JOIN | Join agent to world |
| 0xA4 | SYS_WORLD_LEAVE | Remove agent from world |
| 0xA5 | SYS_WORLD_EVENT | Inject chaos event |
| 0xA6 | SYS_WORLD_STATE | Get world metrics |
| 0xA7 | SYS_WORLD_SNAPSHOT | Save world state |
| 0xA8 | SYS_WORLD_RESTORE | Restore from snapshot |
