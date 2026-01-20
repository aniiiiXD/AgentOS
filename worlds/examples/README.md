# AgentOS World Simulations

Multi-agent simulations demonstrating AgentOS capabilities: process isolation, IPC, state management, and chaos engineering.

## Available Worlds

### 1. Office World

A competitive office simulation where worker agents compete for promotion under a manager agent.

```
┌─────────────────────────────────────────────────────┐
│                    MANAGER AGENT                     │
│  - Assigns tasks                                     │
│  - Evaluates performance                             │
│  - Decides promotions                                │
└───────────────────────┬─────────────────────────────┘
                        │ IPC (tasks, scores)
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Alice   │    │  Bob    │    │ Charlie │  ... (5 workers)
   │ambitious│    │ steady  │    │ slacker │
   └─────────┘    └─────────┘    └─────────┘
```

**Features:**
- 5 worker agents with different personalities (ambitious, team_player, steady, perfectionist, slacker)
- Task assignment and competition
- Point-based scoring system
- Promotion mechanics (Junior → Mid → Senior → Lead)
- Chaos events (server outages, urgent deadlines, etc.)
- LLM-powered decision making (optional)

**Run locally:**
```bash
# Start kernel first
./build/agentos_kernel

# Run simulation
python worlds/examples/office_world/run_office.py

# With LLM-powered workers
python worlds/examples/office_world/run_office.py --use-llm

# Custom worker count
python worlds/examples/office_world/run_office.py --workers 3
```

---

### 2. Systems World

A production infrastructure simulation with servers, SRE agents, and chaos engineering.

```
┌─────────────────────────────────────────────────────┐
│                    SRE AGENT                         │
│  - Monitors metrics                                  │
│  - Detects alerts                                    │
│  - Performs remediation                              │
└───────────────────────┬─────────────────────────────┘
                        │ Metrics + Commands
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ web-1   │    │ api-1   │    │db-primary│
   │ nginx   │    │ app-api │    │ postgres │
   └─────────┘    └─────────┘    └─────────┘
        │               │               │
        └───────── CHAOS INJECTION ─────┘
```

**Features:**
- 6 servers (web, API, database, cache)
- Real-time metrics (CPU, memory, latency, error rate)
- Alert thresholds with warning/critical levels
- 7 chaos scenarios (CPU spike, memory leak, network partition, etc.)
- 6 remediation actions (restart, scale up, failover, etc.)
- SLA tracking
- LLM-assisted diagnosis (optional)

**Run locally:**
```bash
# Start kernel first
./build/agentos_kernel

# Run simulation
python worlds/examples/systems_world/run_systems.py

# High chaos mode
python worlds/examples/systems_world/run_systems.py --chaos-level high

# With LLM-powered diagnosis
python worlds/examples/systems_world/run_systems.py --use-llm
```

---

## Running via Cloud/Relay

Deploy worlds to remote machines using the AgentOS CLI.

### Setup Relay Server

```bash
# Terminal 1: Start relay
cd relay
pip install -r requirements.txt
python relay_server.py
```

### Deploy Kernel to Docker

```bash
# Deploy a Docker kernel
agentos deploy docker --name sim-kernel

# Verify it's connected
agentos status
```

### Run World Remotely

```bash
# Run office simulation on remote kernel
agentos agent run worlds/examples/office_world/run_office.py \
    --machine docker-sim-kernel-xxx

# Run systems simulation on remote kernel
agentos agent run worlds/examples/systems_world/run_systems.py \
    --machine docker-sim-kernel-xxx \
    -- --chaos-level high
```

### Multi-Kernel Deployment

Run different parts of the simulation on different machines:

```bash
# Deploy multiple kernels
agentos deploy docker --name world-control    # Control plane
agentos deploy docker --name world-workers    # Worker agents

# Start control on first kernel
agentos agent run worlds/examples/office_world/manager_agent.py \
    --machine docker-world-control-xxx

# Start workers on second kernel
for name in alice bob charlie diana eve; do
    agentos agent run worlds/examples/office_world/worker_agent.py \
        --machine docker-world-workers-xxx \
        -- --name $name --personality steady
done
```

### Cloud Deployment (AWS/GCP)

```bash
# Deploy to AWS
agentos deploy aws --region us-east-1 --name sim-aws

# Deploy to GCP
agentos deploy gcp --zone us-central1-a --name sim-gcp

# Run simulation on cloud instance
agentos agent run worlds/examples/systems_world/run_systems.py \
    --machine aws-sim-aws-xxx \
    -- --chaos-level extreme --use-llm
```

---

## Using Fleet Client (Python)

Programmatically manage simulations across multiple kernels:

```python
from fleet_client import FleetClient
import asyncio

async def run_distributed_simulation():
    fleet = FleetClient(relay_url="http://localhost:8766")

    # Get all connected machines
    machines = await fleet.get_connected_machines()
    print(f"Found {len(machines)} machines")

    # Deploy simulation to all
    results = await fleet.run_on_all(
        "worlds/examples/office_world/run_office.py",
        args=["--workers", "3"]
    )

    for r in results:
        print(f"{r['machine_id']}: {r['status']}")

    await fleet.close()

asyncio.run(run_distributed_simulation())
```

---

## Customization

### Office World Config

Edit `office_world/office_config.py`:

```python
OFFICE_CONFIG = {
    "rounds": 20,              # Simulation length
    "tasks_per_round": 5,      # More tasks = more competition
    "promotion_rules": {
        "junior_to_mid": 150,  # Adjust difficulty
        ...
    }
}
```

### Systems World Config

Edit `systems_world/systems_config.py`:

```python
SYSTEMS_CONFIG = {
    "rounds": 30,              # Longer simulation
    "sla_target": 99.99,       # Stricter SLA
    "chaos_scenarios": [
        {
            "name": "total_meltdown",
            "probability": 0.02,
            "effect": {"everything_down": True}
        }
    ]
}
```

---

## What This Demonstrates

| Capability | Office World | Systems World |
|------------|--------------|---------------|
| Process Isolation | Each worker is a separate process | Each server is a separate process |
| IPC | Task assignment, scoring | Metrics, remediation commands |
| State Store | Leaderboard, worker registry | Server metrics, alerts |
| Events | Promotion notifications | Chaos injection |
| Fault Tolerance | Workers can crash independently | Servers fail without affecting SRE |
| LLM Integration | Task decisions | Incident diagnosis |
| Multi-Agent Coordination | 6 agents collaborating/competing | 7 agents monitoring/responding |

---

## Troubleshooting

### "Connection refused"
Kernel not running. Start with:
```bash
./build/agentos_kernel
```

### Agents not communicating
Agents need to register names before sending messages:
```python
agent.register_name("my-name")
```

### LLM not working
Set your API key:
```bash
export GEMINI_API_KEY="your-key"
```

### Remote execution fails
Check relay status:
```bash
agentos status
agentos machines list
```
