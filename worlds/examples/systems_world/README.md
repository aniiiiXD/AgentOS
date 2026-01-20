# Systems World Simulation

A production infrastructure simulation with servers, an SRE agent, and chaos engineering. Demonstrates AgentOS for infrastructure monitoring, incident response, and fault tolerance testing.

## Concept

```
                    ┌─────────────────────────┐
                    │       SRE AGENT         │
                    │  - Monitors metrics     │
                    │  - Detects alerts       │
                    │  - Remediates issues    │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│    WEB-1      │       │    API-1      │       │  DB-PRIMARY   │
│    nginx      │       │   app-api     │       │  postgresql   │
│  app-frontend │       │ redis-client  │       │               │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                            │
                    CHAOS INJECTION
              (CPU spikes, memory leaks, partitions)
```

## How It Works

### Infrastructure

| Server | Type | Services | Role |
|--------|------|----------|------|
| web-1 | Web | nginx, app-frontend | Load balancer + frontend |
| web-2 | Web | nginx, app-frontend | Redundant web server |
| api-1 | API | app-api, redis-client | Backend API |
| db-primary | Database | postgresql | Primary database |
| db-replica | Database | postgresql | Replica for failover |
| cache-1 | Cache | redis | In-memory cache |

### Simulation Flow

1. **Servers spawn** and start reporting metrics
2. **SRE agent spawns** and begins monitoring
3. **Each round:**
   - Traffic is generated
   - Chaos events may be injected (based on probability)
   - Servers update their metrics
   - SRE checks for alerts
   - SRE diagnoses and remediates issues
4. **End:** SLA calculated and reported

### Metrics Tracked

| Metric | Warning | Critical | Unit |
|--------|---------|----------|------|
| cpu_usage | 70% | 90% | Percentage |
| memory_usage | 75% | 95% | Percentage |
| error_rate | 1% | 5% | Percentage |
| latency_p99 | 500ms | 2000ms | Milliseconds |
| disk_usage | 80% | 95% | Percentage |

### Chaos Scenarios

| Scenario | Probability | Duration | Effect |
|----------|-------------|----------|--------|
| **cpu_spike** | 15% | 3 rounds | CPU jumps to 95% |
| **memory_leak** | 10% | 5 rounds | Memory grows +10%/round |
| **network_partition** | 5% | 2 rounds | Server isolated |
| **disk_full** | 8% | 4 rounds | Disk at 98% |
| **service_crash** | 12% | 1 round | Random service crashes |
| **traffic_surge** | 10% | 3 rounds | 10x traffic spike |
| **database_slowdown** | 10% | 4 rounds | 5x query latency |

### Remediation Actions

| Action | Success Rate | Cooldown | Description |
|--------|--------------|----------|-------------|
| restart_service | 90% | 2 rounds | Restart crashed service |
| scale_up | 95% | 3 rounds | Add more instances |
| failover | 85% | 1 round | Switch to backup |
| clear_cache | 98% | 1 round | Free memory by clearing cache |
| rate_limit | 95% | 1 round | Enable rate limiting |
| rollback | 80% | 5 rounds | Rollback to previous version |

### SRE Skill Levels

| Level | Diagnosis Accuracy | Remediation Speed | Panic Threshold |
|-------|-------------------|-------------------|-----------------|
| junior | 60% | 50% | 2 alerts |
| mid | 80% | 70% | 4 alerts |
| senior | 95% | 90% | 8 alerts |

## Usage

### Local Execution

```bash
# Terminal 1: Start kernel
./build/agentos_kernel

# Terminal 2: Run simulation
cd /home/anixd/Documents/AGENTOS
python worlds/examples/systems_world/run_systems.py
```

### Options

```bash
# Chaos levels: low, medium, high, extreme
python worlds/examples/systems_world/run_systems.py --chaos-level high

# Enable LLM for SRE diagnosis
python worlds/examples/systems_world/run_systems.py --use-llm

# Quiet mode
python worlds/examples/systems_world/run_systems.py --quiet

# Combined: extreme chaos with LLM
python worlds/examples/systems_world/run_systems.py \
    --chaos-level extreme --use-llm
```

### Chaos Level Multipliers

| Level | Probability Multiplier | Description |
|-------|------------------------|-------------|
| low | 0.5x | Calm day, few incidents |
| medium | 1.0x | Normal operations |
| high | 2.0x | Bad day, many incidents |
| extreme | 3.0x | Everything is on fire |

### Remote Execution (via Relay)

```bash
# Deploy kernel to Docker
agentos deploy docker --name infra-sim

# Run simulation remotely
agentos agent run worlds/examples/systems_world/run_systems.py \
    --machine docker-infra-sim-xxx

# With chaos and LLM
agentos agent run worlds/examples/systems_world/run_systems.py \
    --machine docker-infra-sim-xxx \
    -- --chaos-level extreme --use-llm
```

### Cloud Deployment

```bash
# Deploy to AWS for more realistic simulation
agentos deploy aws --region us-east-1 --name prod-chaos-test

# Run extreme chaos test
agentos agent run worlds/examples/systems_world/run_systems.py \
    --machine aws-prod-chaos-test-xxx \
    -- --chaos-level extreme
```

### Run Individual Agents

```bash
# Terminal 1: Start a server manually
python worlds/examples/systems_world/server_agent.py --name web-1

# Terminal 2: Start another server
python worlds/examples/systems_world/server_agent.py --name db-primary

# Terminal 3: Start SRE
python worlds/examples/systems_world/sre_agent.py \
    --name oncall --skill senior --use-llm
```

## Example Output

```
======================================================================
SYSTEMS WORLD SIMULATION
Infrastructure: CloudInfra Production
Servers: 6
Rounds: 15
Chaos Level: high (2.0x)
LLM Mode: OFF
======================================================================

Spawning servers...
  [WEB] web-1: PID 12345
  [WEB] web-2: PID 12346
  [API] api-1: PID 12347
  [DATABASE] db-primary: PID 12348
  [DATABASE] db-replica: PID 12349
  [CACHE] cache-1: PID 12350

Spawning SRE...
  SRE oncall: PID 12351

======================================================================
SIMULATION RUNNING
======================================================================

--- Round 1 ---
    web-1: OK (health=100, cpu=25%)
    web-2: OK (health=100, cpu=22%)
    api-1: OK (health=100, cpu=30%)
    db-primary: OK (health=100, cpu=18%)
    db-replica: OK (health=100, cpu=15%)
    cache-1: OK (health=100, cpu=12%)

--- Round 4 ---
  CHAOS: cpu_spike -> api-1
    web-1: OK (health=100, cpu=28%)
    web-2: OK (health=95, cpu=35%)
    api-1: CRIT (health=40, cpu=95%)
    db-primary: OK (health=100, cpu=22%)

[SRE oncall] ALERT [CRITICAL] api-1: cpu_usage=95
[SRE oncall] Executing scale_up on api-1...
[SRE oncall] Remediation succeeded: scale_up on api-1

--- Round 7 ---
  CHAOS: memory_leak -> db-primary
  CHAOS: traffic_surge -> web-1
    web-1: WARN (health=70, cpu=82%)
    api-1: OK (health=90, cpu=45%)
    db-primary: WARN (health=75, cpu=40%)

[SRE oncall] ALERT [WARNING] web-1: cpu_usage=82
[SRE oncall] ALERT [WARNING] db-primary: memory_usage=78
[SRE oncall] Executing rate_limit on web-1...
[SRE oncall] Executing clear_cache on db-primary...

...

======================================================================
SIMULATION COMPLETE
======================================================================

Infrastructure Summary:
  Rounds: 15
  Chaos Level: high
  Availability: 94.67%
  SLA Target: 99.9%

  SLA MISSED by 5.23%

Shutting down...
Simulation complete!
```

## Files

| File | Description |
|------|-------------|
| `systems_config.py` | Configuration (servers, chaos, thresholds) |
| `server_agent.py` | Server agent (reports metrics, handles chaos) |
| `sre_agent.py` | SRE agent (monitors, diagnoses, remediates) |
| `run_systems.py` | Orchestrator (spawns agents, injects chaos) |

## Customization

### Add New Server

Edit `systems_config.py`:
```python
"servers": [
    ...
    {
        "name": "queue-1",
        "type": "queue",
        "cpu_capacity": 50,
        "memory_capacity": 8192,
        "services": ["rabbitmq"],
        "health": 100,
    },
]
```

### Add New Chaos Scenario

```python
"chaos_scenarios": [
    ...
    {
        "name": "dns_failure",
        "description": "DNS resolution fails",
        "probability": 0.05,
        "duration": 2,
        "effect": {"dns_down": True},
    },
]
```

### Adjust SLA Target

```python
SYSTEMS_CONFIG = {
    "sla_target": 99.99,  # Stricter SLA (four nines)
    ...
}
```

### Add New Remediation

```python
"remediation_actions": [
    ...
    {
        "name": "blue_green_deploy",
        "description": "Switch to standby environment",
        "success_rate": 0.95,
        "cooldown": 10,
    },
]
```

## What This Demonstrates

| AgentOS Feature | How It's Used |
|-----------------|---------------|
| **Process Isolation** | Each server is a separate process (can crash independently) |
| **IPC (send/recv)** | Metrics reporting, remediation commands |
| **State Store** | Server metrics with TTL |
| **Agent Naming** | `server-{name}`, `sre-{name}` for routing |
| **LLM Integration** | SRE diagnosis reasoning |
| **Fault Tolerance** | Servers fail without affecting SRE agent |
| **Real-time Monitoring** | Metrics polled and evaluated each round |

## Use Cases

1. **Chaos Engineering Training**: Test how systems behave under failure
2. **SRE Training**: Practice incident response in safe environment
3. **LLM Evaluation**: Compare LLM vs rule-based diagnosis
4. **Architecture Testing**: Validate redundancy and failover
5. **Benchmark**: Measure MTTR (mean time to resolve) under different conditions
