# Clove Syscall Reference

Note: LLM calls are handled by the SDK via `agents/llm_service`. The kernel no longer processes `SYS_THINK`; it returns an error stub.

## Wire Protocol

```
┌──────────────┬──────────────┬─────────┬───────────────┐
│  Magic (4B)  │ Agent ID (4B)│ Op (1B) │ Payload Len   │
│  0x41474E54  │   uint32     │  uint8  │   uint64 (8B) │
└──────────────┴──────────────┴─────────┴───────────────┘
                    17 bytes total, then payload
```

- **Magic**: `0x41474E54` ("AGNT")
- **Max payload**: 1 MB
- **Socket**: `/tmp/clove.sock`

---

## Syscall Table

### Core

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x00` | NOOP | `string` | Same string (echo) |
| `0xFE` | HELLO | `{}` | `{"success", "protocol_version", "kernel_version", "features"}` |
| `0xFF` | EXIT | — | Acknowledgment |

### LLM

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x01` | THINK | `{"prompt", "image?", "system_instruction?", "thinking_level?", "temperature?", "model?"}` | Error stub (kernel-disabled) |
| `0xF0` | LLM_REPORT | `{"tokens", "success?"}` | `{"success", "tokens", "quota_exceeded", "error?"}` |

### Filesystem

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x02` | EXEC | `{"command", "cwd?", "timeout?", "async?", "request_id?"}` | `{"success", "stdout", "stderr", "exit_code"}` or `{"success", "async", "request_id", "status"}` |
| `0x03` | READ | `{"path"}` | `{"success", "content", "size"}` |
| `0x04` | WRITE | `{"path", "content", "mode?"}` | `{"success", "bytes_written"}` |

Note: If `async` is omitted for EXEC or HTTP, the kernel defaults to async execution.

### Agent Management

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x10` | SPAWN | `{"name", "script", "sandboxed?", "network?", "limits?", "restart_policy?", "max_restarts?", "restart_window?"}` | `{"success", "agent_id", "pid", "restart_policy"}` |
| `0x11` | KILL | `{"name"}` or `{"id"}` | `{"killed", "agent_id"}` |
| `0x12` | LIST | — | `[{"id", "name", "pid", "state", "uptime_ms"}]` |
| `0x14` | PAUSE | `{"name"}` or `{"id"}` | `{"success", "paused", "agent_id"}` |
| `0x15` | RESUME | `{"name"}` or `{"id"}` | `{"success", "resumed", "agent_id"}` |

**Restart configuration:**
- `restart_policy`: `"never"` (default), `"on-failure"`, or `"always"`
- `max_restarts`: Maximum restarts allowed within `restart_window` (default: 5)
- `restart_window`: Time window in seconds for counting restarts (default: 300)

### IPC (Inter-Agent Communication)

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x20` | SEND | `{"to" or "to_name", "message"}` | `{"success", "delivered_to"}` |
| `0x21` | RECV | `{"max?"}` | `{"success", "count", "messages"}` |
| `0x22` | BROADCAST | `{"message", "include_self?"}` | `{"success", "delivered_count"}` |
| `0x23` | REGISTER | `{"name"}` | `{"success", "agent_id", "name"}` |

### Permissions

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x40` | GET_PERMS | — | `{"success", "permissions"}` |
| `0x41` | SET_PERMS | `{"agent_id?", "level?" or "permissions?"}` | `{"success", "agent_id"}` |

**Levels**: `unrestricted`, `standard`, `sandboxed`, `readonly`, `minimal`

### State Store

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x30` | STORE | `{"key", "value", "scope?", "ttl?"}` | `{"success", "key"}` |
| `0x31` | FETCH | `{"key"}` | `{"success", "exists", "value", "scope"}` |
| `0x32` | DELETE | `{"key"}` | `{"success", "deleted"}` |
| `0x33` | KEYS | `{"prefix?"}` | `{"success", "keys", "count"}` |

**Scopes**: `global` (all agents), `agent` (private), `session` (until restart)

### Network

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x50` | HTTP | `{"url", "method?", "headers?", "body?", "timeout?", "async?", "request_id?"}` | `{"success", "status_code", "body"}` or `{"success", "async", "request_id", "status"}` |

### Events (Pub/Sub)

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x60` | SUBSCRIBE | `{"event_types": [...]}` | `{"success", "subscribed": [...]}` |
| `0x61` | UNSUBSCRIBE | `{"event_types": [...]}` | `{"success", "unsubscribed": [...]}` |
| `0x62` | POLL_EVENTS | `{"max?"}` | `{"success", "events": [...], "count"}` |
| `0x63` | EMIT | `{"event_type", "data"}` | `{"success", "delivered_to"}` |

**Event types**: `AGENT_SPAWNED`, `AGENT_EXITED`, `AGENT_RESTARTING`, `AGENT_ESCALATED`, `MESSAGE_RECEIVED`, `STATE_CHANGED`, `SYSCALL_BLOCKED`, `RESOURCE_WARNING`, `CUSTOM`

**Restart events:**
- `AGENT_RESTARTING`: Emitted when an agent is being auto-restarted. Data: `{"agent_name", "restart_count", "exit_code"}`
- `AGENT_ESCALATED`: Emitted when an agent exceeds `max_restarts` within the restart window. Data: `{"agent_name", "restart_count", "exit_code"}`

**Event structure**: `{"type", "data", "source_agent_id", "age_ms"}`

### World Simulation

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0xA0` | WORLD_CREATE | `{"name", "config"}` | `{"success", "world_id"}` |
| `0xA1` | WORLD_DESTROY | `{"world_id"}` | `{"success"}` |
| `0xA2` | WORLD_LIST | — | `{"success", "worlds": [...]}` |
| `0xA3` | WORLD_JOIN | `{"world_id", "agent_id"}` | `{"success"}` |
| `0xA4` | WORLD_LEAVE | `{"world_id", "agent_id"}` | `{"success"}` |
| `0xA5` | WORLD_EVENT | `{"world_id", "event_type", "config"}` | `{"success"}` |
| `0xA6` | WORLD_STATE | `{"world_id"}` | `{"success", "state"}` |
| `0xA7` | WORLD_SNAPSHOT | `{"world_id", "path?"}` | `{"success", "snapshot_path"}` |
| `0xA8` | WORLD_RESTORE | `{"world_id", "snapshot_path"}` | `{"success"}` |

**Chaos events**: `network_partition`, `latency_spike`, `packet_loss`, `agent_crash`, `resource_pressure`

### Remote Tunnel

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0xB0` | TUNNEL_CONNECT | `{"relay_url", "token"}` | `{"success", "tunnel_id"}` |
| `0xB1` | TUNNEL_DISCONNECT | — | `{"success"}` |
| `0xB2` | TUNNEL_STATUS | — | `{"success", "connected", "relay_url"}` |
| `0xB3` | TUNNEL_LIST_REMOTES | — | `{"success", "remotes": [...]}` |
| `0xB4` | TUNNEL_CONFIG | `{"config"}` | `{"success"}` |

### Metrics

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0xC0` | METRICS_SYSTEM | — | `{"success", "metrics": {"cpu", "memory", "disk", "network"}}` |
| `0xC1` | METRICS_AGENT | `{"agent_id?"}` | `{"success", "metrics": {"agent_id", "name", "process", "cgroup"}}` |
| `0xC2` | METRICS_ALL_AGENTS | — | `{"success", "agents": [...], "count"}` |
| `0xC3` | METRICS_CGROUP | `{"cgroup_path?"}` | `{"success", "metrics": {"cpu", "memory", "pids"}}` |

**System metrics structure:**
```json
{
  "cpu": {"percent", "per_core": [...], "count", "load_avg": [1m, 5m, 15m]},
  "memory": {"total", "available", "used", "percent"},
  "disk": {"read_bytes", "write_bytes"},
  "network": {"bytes_sent", "bytes_recv"}
}
```

**Agent metrics structure:**
```json
{
  "agent_id": 123,
  "name": "worker",
  "status": "running",
  "uptime_ms": 12345,
  "process": {
    "pid": 456,
    "cpu": {"percent": 5.2},
    "memory": {"rss": 1024000, "vms": 2048000},
    "threads": 4,
    "fds": 12
  },
  "cgroup": {"valid": true, "cpu_usage_usec": 123456, "mem_current": 1024000}
}
```

### Audit Logging

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x76` | GET_AUDIT_LOG | `{"category?", "agent_id?", "since_id?", "limit?"}` | `{"success", "count", "entries": [...]}` |
| `0x77` | SET_AUDIT_CONFIG | `{"log_syscalls?", "log_security?", "log_lifecycle?", ...}` | `{"success", "config"}` |

**Audit categories:**
- `SECURITY` - Permission denied, blocked commands
- `AGENT_LIFECYCLE` - Spawn, kill, pause, resume
- `IPC` - Send, recv, broadcast
- `STATE_STORE` - Store, fetch, delete
- `RESOURCE` - Quota exceeded, resource warnings
- `SYSCALL` - All syscalls (verbose)
- `NETWORK` - HTTP requests, tunnel events
- `WORLD` - World simulation events

**Audit entry structure:**
```json
{
  "id": 123,
  "timestamp": "2026-01-26T12:00:00.000Z",
  "category": "AGENT_LIFECYCLE",
  "event_type": "AGENT_SPAWNED",
  "agent_id": 1,
  "agent_name": "worker",
  "success": true,
  "details": {...}
}
```

### Async Results

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x80` | ASYNC_POLL | `{"max?"}` | `{"success", "results": [{"request_id", "opcode", "opcode_name", "payload"}], "count"}` |

**Async flow (EXEC/HTTP/THINK):**
1) Send syscall with `"async": true` (and optional `"request_id"`).
2) Kernel returns `{"success": true, "async": true, "request_id", "status": "accepted"}`.
3) Poll results with `ASYNC_POLL`. Each result’s `payload` is the JSON string of the normal sync response.

### Execution Recording & Replay

| Op | Name | Payload | Response |
|----|------|---------|----------|
| `0x70` | RECORD_START | `{"include_think?", "include_http?", "include_exec?", "filter_agents?", "max_entries?"}` | `{"success", "state"}` |
| `0x71` | RECORD_STOP | — | `{"success", "entry_count"}` |
| `0x72` | RECORD_STATUS | `{"export?"}` | `{"success", "state", "entry_count", "recording?"}` |
| `0x73` | REPLAY_START | `{"recording"}` | `{"success", "total_entries"}` |
| `0x74` | REPLAY_STATUS | — | `{"success", "state", "progress", "entries_replayed", "entries_skipped", "error?"}` |

**Recording configuration:**
- `include_think`: Include LLM calls (non-deterministic, default: false)
- `include_http`: Include HTTP calls (non-deterministic, default: false)
- `include_exec`: Include exec calls (may be non-deterministic, default: false)
- `filter_agents`: Only record these agent IDs (empty = all)
- `max_entries`: Maximum entries to keep in buffer (default: 50000)

**Recording states:** `IDLE`, `RECORDING`, `PAUSED`

**Replay states:** `IDLE`, `RUNNING`, `PAUSED`, `COMPLETED`, `ERROR`

**Execution log entry structure:**
```json
{
  "sequence_id": 1,
  "timestamp": "2026-01-26T12:00:00.000Z",
  "agent_id": 1,
  "opcode": 16,
  "opcode_name": "SPAWN",
  "payload": "{\"name\": \"worker\", ...}",
  "response": "{\"success\": true, ...}",
  "duration_us": 1234,
  "success": true
}
```

**Usage notes:**
- By default, non-deterministic syscalls (THINK, HTTP, EXEC) are excluded from recording
- Deterministic syscalls are replayed to recreate system state
- Use `export: true` in RECORD_STATUS to get the full recording for later replay
- Pass the exported recording JSON string to REPLAY_START to begin replay

---

## Future Syscalls

### Network Extended

| Op | Name | Description | Status |
|----|------|-------------|--------|
| `0x51` | DOWNLOAD | Download file from URL to path | Planned |

### Task Orchestration

| Op | Name | Description | Status |
|----|------|-------------|--------|
| `0x80` | TASK_CREATE | Create orchestrated task | Planned |
| `0x81` | TASK_ASSIGN | Assign agent to task | Planned |
| `0x82` | TASK_STATUS | Get task status | Planned |
| `0x83` | TASK_COMPLETE | Mark task complete | Planned |
| `0x84` | TASK_LIST | List active tasks | Planned |

### Resource Management

| Op | Name | Description | Status |
|----|------|-------------|--------|
| `0x92` | SET_QUOTA | Set resource quota for agent | Planned |

---

## Cloud Deployment (Phase 8)

Phase 8 added cloud deployment infrastructure without new kernel syscalls. The deployment system operates at the orchestration layer:

- **CLI Tool** (`clove`): Fleet management from terminal
- **REST API** (`relay/api.py`): Endpoints for machine/agent/token management
- **Docker Deployment**: Containerized kernels via `deploy/docker/`
- **AWS Deployment**: EC2 provisioning via `deploy/terraform/aws/`
- **GCP Deployment**: Compute Engine via `deploy/terraform/gcp/`

See [CLI Reference](../cli/README.md) for usage.

---

## Status Codes

| Code | Name |
|------|------|
| `0x00` | OK |
| `0x01` | ERROR |
| `0x02` | INVALID_MSG |
| `0x03` | NOT_FOUND |
| `0x04` | TIMEOUT |

---

## Python SDK

```python
from clove_sdk import CloveClient

with CloveClient() as c:
    c.noop("ping")                              # NOOP
    c.think("What is 2+2?")                     # THINK
    c.exec("ls -la")                            # EXEC
    c.read_file("/tmp/test.txt")                # READ
    c.write_file("/tmp/out.txt", "data")        # WRITE
    c.spawn("worker", "/path/to/agent.py",       # SPAWN
            restart_policy="on-failure",
            max_restarts=3)
    c.kill(name="worker")                       # KILL
    c.list_agents()                             # LIST
    c.register_name("orchestrator")             # REGISTER
    c.send_message({"task": "go"}, to_name="worker")  # SEND
    c.recv_messages()                                  # RECV
    c.broadcast({"event": "done"})              # BROADCAST
    c.get_permissions()                         # GET_PERMS
    c.set_permissions(level="sandboxed")        # SET_PERMS
    c.store("key", {"value": 123})              # STORE
    c.fetch("key")                              # FETCH
    c.delete_key("key")                         # DELETE
    c.list_keys("prefix:")                      # KEYS
    c.http("https://api.example.com/data")      # HTTP
    c.subscribe(["AGENT_SPAWNED", "AGENT_RESTARTING", "CUSTOM"])  # SUBSCRIBE
    c.poll_events()                             # POLL_EVENTS
    c.emit_event("CUSTOM", {"msg": "hello"})    # EMIT
    c.unsubscribe(["CUSTOM"])                   # UNSUBSCRIBE
    c.get_system_metrics()                      # METRICS_SYSTEM
    c.get_agent_metrics(agent_id=123)           # METRICS_AGENT
    c.get_all_agent_metrics()                   # METRICS_ALL_AGENTS
    c.get_cgroup_metrics()                      # METRICS_CGROUP
    c.pause(name="worker")                      # PAUSE
    c.resume(name="worker")                     # RESUME
    c.get_audit_log(category="AGENT_LIFECYCLE") # GET_AUDIT_LOG
    c.set_audit_config(log_lifecycle=True)      # SET_AUDIT_CONFIG
    c.start_recording()                         # RECORD_START
    c.stop_recording()                          # RECORD_STOP
    status = c.get_recording_status(export=True)  # RECORD_STATUS
    c.start_replay(status["recording"])         # REPLAY_START
    c.get_replay_status()                       # REPLAY_STATUS
```
