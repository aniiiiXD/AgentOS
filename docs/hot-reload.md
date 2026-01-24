# Hot Reload & Auto-Recovery

When an agent crashes, Clove automatically restarts it with exponential backoff.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Clove Kernel                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 Agent Supervisor                          │   │
│  │  - Monitors agent health via process status              │   │
│  │  - Detects crashes (exit codes, zombie processes)        │   │
│  │  - Triggers auto-restart with exponential backoff        │   │
│  │  - Escalates after max_restarts exceeded                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
│  │  Agent A    │      │  Agent B    │      │  Agent C    │     │
│  │  running    │      │  crashed    │      │  running    │     │
│  └─────────────┘      └──────┬──────┘      └─────────────┘     │
│                              │                                   │
│                              ▼                                   │
│                       ┌─────────────┐                           │
│                       │  Agent B    │                           │
│                       │  restarted  │  <- Auto-restarted        │
│                       │  (attempt 1)│     with 1s backoff       │
│                       └─────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

```python
from clove_sdk import CloveClient

with CloveClient() as client:
    client.spawn(
        name="worker",
        script="/path/to/agent.py",
        restart_policy="on-failure",  # always | on-failure | never
        max_restarts=5,               # max restarts in restart_window
        restart_window=300,           # seconds (default: 5 minutes)
    )
```

## Restart Policies

| Policy | Behavior |
|--------|----------|
| `never` | Never restart (default) |
| `on-failure` | Restart only on non-zero exit code |
| `always` | Always restart regardless of exit code |

## Backoff Strategy

Restart delay increases exponentially to prevent restart storms:

```
Attempt 1: 1s delay
Attempt 2: 2s delay
Attempt 3: 4s delay
Attempt 4: 8s delay
...
Maximum:   60s delay
```

After `max_restarts` within `restart_window`, the agent is **escalated** — it stops restarting and emits an `AGENT_ESCALATED` event.

## Events

Subscribe to restart events for monitoring:

```python
client.subscribe(["AGENT_RESTARTING", "AGENT_ESCALATED"])

events = client.poll_events()
for event in events.get("events", []):
    if event["type"] == "AGENT_RESTARTING":
        print(f"Agent restarting: {event['data']}")
    elif event["type"] == "AGENT_ESCALATED":
        print(f"Agent escalated (giving up): {event['data']}")
```

## Implementation Details

- **Crash detection**: Automatic via `waitpid()` on each reactor poll cycle
- **Process cleanup**: Zombie processes reaped immediately
- **State preservation**: Agent ID preserved across restarts
- **Resource limits**: Same cgroup limits applied to restarted process

## Future Enhancements

- State checkpointing — periodic snapshots of agent state to restore on crash
- Health checks — heartbeat protocol for detecting unresponsive agents
- Hot code reload — update agent code without full restart
