# Office World Simulation

A multi-agent simulation where workers compete for promotion under a manager's supervision. Demonstrates AgentOS process isolation, IPC, state management, and LLM integration.

## Concept

```
                    ┌─────────────────────────┐
                    │      MANAGER AGENT      │
                    │  - Assigns tasks        │
                    │  - Tracks performance   │
                    │  - Decides promotions   │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │    ALICE      │   │     BOB       │   │   CHARLIE     │
    │  (ambitious)  │   │   (steady)    │   │   (slacker)   │
    │  Takes risks  │   │  Plays safe   │   │  Low effort   │
    └───────────────┘   └───────────────┘   └───────────────┘
            │                   │                   │
            └───────────────────┴───────────────────┘
                        COMPETITION
                    (tasks, points, ranks)
```

## How It Works

### Simulation Flow

1. **Manager spawns** and waits for workers to register
2. **Workers spawn** with assigned personalities
3. **Each round:**
   - Manager generates 3 random tasks
   - Workers decide which task to claim (based on personality)
   - Workers complete tasks (success based on work ethic)
   - Manager awards points based on completion quality
4. **Every 3 rounds:** Promotion review
5. **End:** Winner announced based on total points

### Worker Personalities

| Personality | Risk Taking | Collaboration | Work Ethic | Strategy |
|-------------|-------------|---------------|------------|----------|
| **ambitious** | 80% | 30% | 90% | Takes hard tasks, rarely helps others |
| **team_player** | 50% | 90% | 70% | Helps teammates, moderate tasks |
| **steady** | 30% | 50% | 80% | Prefers easy tasks, consistent |
| **perfectionist** | 60% | 40% | 100% | Always completes, high quality |
| **slacker** | 20% | 60% | 40% | Avoids work, sometimes fails |

### Task Types

| Task | Difficulty | Points | Description |
|------|------------|--------|-------------|
| code_review | Easy | 10 | Review a pull request |
| documentation | Easy | 15 | Write module docs |
| bug_fix | Medium | 25 | Fix a reported bug |
| client_meeting | Medium | 30 | Attend and summarize meeting |
| optimization | Hard | 40 | Optimize slow component |
| feature_implementation | Hard | 50 | Implement new feature |

### Promotion Ladder

```
Junior (start) → Mid-Level (100 pts) → Senior (300 pts) → Team Lead (600 pts)
```

### Chaos Events

Random events that affect the simulation:

- **Server Outage** (10%): All tasks paused for 1 round
- **Urgent Deadline** (15%): Double points but must complete fast
- **Team Lunch** (5%): Everyone gets +5 morale
- **Coffee Machine Broken** (10%): Productivity -20%

## Usage

### Local Execution

```bash
# Terminal 1: Start kernel
./build/agentos_kernel

# Terminal 2: Run simulation
cd /home/anixd/Documents/AGENTOS
python worlds/examples/office_world/run_office.py
```

### Options

```bash
# Custom number of workers (max 8)
python worlds/examples/office_world/run_office.py --workers 3

# Enable LLM for worker decisions
python worlds/examples/office_world/run_office.py --use-llm

# Quiet mode (less output)
python worlds/examples/office_world/run_office.py --quiet

# Combined
python worlds/examples/office_world/run_office.py --workers 5 --use-llm
```

### Remote Execution (via Relay)

```bash
# Deploy kernel to Docker
agentos deploy docker --name office-sim

# Run simulation remotely
agentos agent run worlds/examples/office_world/run_office.py \
    --machine docker-office-sim-xxx

# With arguments
agentos agent run worlds/examples/office_world/run_office.py \
    --machine docker-office-sim-xxx \
    -- --workers 5 --use-llm
```

### Run Individual Agents

You can run agents separately for debugging:

```bash
# Terminal 1: Start a worker manually
python worlds/examples/office_world/worker_agent.py \
    --name alice --personality ambitious

# Terminal 2: Start another worker
python worlds/examples/office_world/worker_agent.py \
    --name bob --personality steady

# Terminal 3: Start manager (after workers)
python worlds/examples/office_world/manager_agent.py
```

## Example Output

```
======================================================================
OFFICE WORLD SIMULATION
Office: TechCorp Office
Workers: 5
Rounds: 10
LLM Mode: OFF
======================================================================

Spawning workers...
  Spawned alice (ambitious): PID 12345
  Spawned bob (team_player): PID 12346
  Spawned charlie (steady): PID 12347
  Spawned diana (perfectionist): PID 12348
  Spawned eve (slacker): PID 12349

Spawning manager...
  Spawned manager: PID 12350

======================================================================
SIMULATION RUNNING
======================================================================

[Round 1] Leaderboard:
  1. diana [Junior]: 45 pts
  2. alice [Junior]: 40 pts
  3. bob [Junior]: 25 pts

[Round 3] Leaderboard:
  1. alice [Junior]: 95 pts
  2. diana [Junior]: 90 pts
  3. charlie [Junior]: 65 pts

========================================
PROMOTION REVIEW
========================================
  PROMOTED: alice (Junior -> Mid-Level)

...

======================================================================
FINAL RESULTS
======================================================================

Final Standings:
  1. alice [Senior]: 385 pts [WINNER]
  2. diana [Mid-Level]: 290 pts [2nd]
  3. bob [Mid-Level]: 215 pts [3rd]
  4. charlie [Junior]: 180 pts
  5. eve [Junior]: 95 pts

Simulation complete!
```

## Files

| File | Description |
|------|-------------|
| `office_config.py` | Configuration (tasks, personalities, rules) |
| `manager_agent.py` | Manager agent (assigns tasks, promotes) |
| `worker_agent.py` | Worker agent (competes for promotion) |
| `run_office.py` | Orchestrator (spawns agents, runs simulation) |

## Customization

### Change Simulation Length

Edit `office_config.py`:
```python
OFFICE_CONFIG = {
    "rounds": 20,  # More rounds
    "tasks_per_round": 5,  # More tasks per round
    ...
}
```

### Add New Personality

```python
WORKER_PERSONALITIES = {
    "workaholic": {
        "risk_tolerance": 0.7,
        "collaboration": 0.2,
        "work_ethic": 1.0,
    },
    ...
}
```

### Adjust Promotion Difficulty

```python
"promotion_rules": {
    "junior_to_mid": 200,   # Harder to promote
    "mid_to_senior": 500,
    "senior_to_lead": 1000,
}
```

## What This Demonstrates

| AgentOS Feature | How It's Used |
|-----------------|---------------|
| **Process Isolation** | Each worker is a separate OS process |
| **IPC (send/recv)** | Task assignment, completion reports |
| **State Store** | Leaderboard, worker registry |
| **Agent Naming** | `register_name()` for discovery |
| **Broadcast** | Manager announces to all workers |
| **LLM Integration** | Optional task decision reasoning |
| **Spawn/Kill** | Dynamic agent lifecycle |
