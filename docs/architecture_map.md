# Clove Architecture Map (Deep)

This document maps the major subsystems (kernel, runtime, worlds, relay, SDK) and their data flows. It reflects the current layout where LLM calls are handled by the SDK via `agents/llm_service`.

## 1) High-Level Topology

```
Python SDK (local)                          Relay (optional)
┌──────────────────────┐                  ┌──────────────────────┐
│ clove_sdk.client     │◄────WebSocket────│ relay/*              │
│ clove_sdk.remote     │                  └──────────────────────┘
└─────────┬────────────┘
          │ Unix socket (/tmp/clove.sock)
          ▼
┌───────────────────────────────────────────────────────────────┐
│                        Clove Kernel (C++)                      │
│  Reactor + SocketServer + SyscallRouter + Modules              │
└───────────────────────────────────────────────────────────────┘
          │
          ├─► AgentManager -> AgentProcess -> Sandbox (namespaces/cgroups)
          ├─► MetricsCollector (/proc, /sys)
          ├─► StateStore / EventBus / Audit / Execution logs
          └─► TunnelClient (python subprocess to relay)

LLM path (SDK-local):
clove_sdk.client.think() -> clove_sdk.llm_service -> agents/llm_service.py -> Gemini API
```

## 2) Kernel Core

### Kernel (`src/kernel/kernel.*`)
- Owns subsystem instances and a shared `KernelContext`.
- Registers syscall modules into `SyscallRouter`.
- Runs event loop (`Reactor`) and dispatches socket events.
- Emits lifecycle and restart events.

### Context (`src/kernel/context.hpp`)
- Immutable references to all subsystems; passed to syscall modules.

### Reactor (`src/kernel/reactor.*`)
- Epoll-based event loop.
- Manages server socket and client sockets; drives non-blocking IO.

### SocketServer (`src/ipc/transport/socket_server.*`)
- Accepts Unix socket connections.
- Parses `MessageHeader` + payload into `ipc::Message`.
- Invokes kernel handler, queues responses.

### SyscallRouter (`src/kernel/syscall_router.*`)
- Dispatch table mapping opcode → handler.
- Unknown opcode returns payload unchanged with a warning.

## 3) Syscall Modules (Kernel)

All modules live in `src/kernel/syscalls/*.cpp` and are registered in `Kernel::Kernel()`:

- **AgentSyscalls**: spawn/kill/list/pause/resume (`AgentManager`)
- **ExecSyscalls**: shell exec via `popen` (permission-gated)
- **FileSyscalls**: read/write; supports virtual filesystem when agent is in a world
- **IpcSyscalls**: register/send/recv/broadcast via `AgentMailboxRegistry`
- **AsyncSyscalls**: poll results from `AsyncTaskManager`
- **NetworkSyscalls**: HTTP via `curl` (permission-gated), supports world network intercepts
- **PermissionSyscalls**: get/set permissions and levels (`PermissionsStore`)
- **StateSyscalls**: key/value storage (`StateStore`)
- **EventSyscalls**: pub/sub over `EventBus`
- **AuditSyscalls**: read/config audit logs (`AuditLogger`)
- **ReplaySyscalls**: execution recording & replay (`ExecutionLogger`)
- **MetricsSyscalls**: system/agent/cgroup metrics (`MetricsCollector`)
- **WorldSyscalls**: create/destroy/list/join/leave worlds (`WorldEngine`)
- **TunnelSyscalls**: relay remote agent syscalls via `TunnelClient`
- **LlmSyscalls**: stubbed; returns error (LLM moved out of kernel)

### AsyncTaskManager (`src/kernel/async_task_manager.*`)
- Worker pool that executes async syscall tasks.
- Results are keyed per agent and polled by `SYS_ASYNC_POLL`.

## 4) Runtime Layer

### AgentManager (`src/runtime/agent/manager.*`)
- Spawns and tracks `AgentProcess` instances.
- Handles restart policies with backoff and escalation.
- Emits restart events for observers.

### AgentProcess (`src/runtime/agent/process.*`)
- Represents one agent process.
- Manages start/stop/pause/resume.
- Tracks per-agent metrics (LLM counts remain but are SDK-local now).

### Sandbox (`src/runtime/sandbox/*`)
- Uses Linux namespaces + cgroups v2 for isolation.
- Reports `IsolationStatus` indicating degraded isolation.

## 5) Worlds

### WorldEngine (`src/worlds/world_engine.*`)
- Manages worlds and per-agent membership.
- Provides virtual filesystem/network mocks and chaos injection.

### VirtualFS (`src/worlds/virtual_fs.*`)
- In-memory filesystem layer for deterministic test worlds.

## 6) Metrics & Logs

### MetricsCollector (`src/metrics/metrics.*`)
- Reads `/proc` and `/sys` to report CPU/mem/disk/net.
- Provides process and cgroup metrics for agents.

### AuditLogger / ExecutionLogger (`src/kernel/audit_log.*`, `execution_log.*`)
- Audit: syscalls and security events.
- Execution: record/replay capability with filtering.

## 7) Tunnel + Relay (Remote)

### TunnelClient (`src/services/tunnel/*`)
- Spawns `scripts/tunnel_client.py` and uses JSON over stdin/stdout.
- Receives remote agent syscalls and forwards to kernel.
- Sends syscall responses back to relay.

### Relay (`relay/*`)
- WebSocket hub for kernels and remote agents.
- Auth + routing; delivers syscalls/responses over relay.

## 8) Python SDK

### Local SDK (`agents/python_sdk/clove_sdk/client.py`)
- Unix socket client for kernel syscalls.
- `think()` now uses local LLM service wrapper, not the kernel.

### Remote SDK (`agents/python_sdk/clove_sdk/remote.py`)
- WebSocket client to relay; forwards syscalls to kernel.
- `think()` is also SDK-local to avoid kernel dependency.

### LLM wrapper (`agents/python_sdk/clove_sdk/llm_service.py`)
- Runs `agents/llm_service/llm_service.py` as a subprocess per call.
- Returns JSON response (content/tokens/function calls).

### Agentic loop (`agents/python_sdk/clove_sdk/agentic.py`)
- Generic tool-using loop; uses `CloveClient` syscalls for tools.

## 9) Key Data Flows

### Syscall Flow (local)
```
SDK -> Unix socket -> SocketServer -> SyscallRouter -> Module -> Response
```

### Syscall Flow (remote)
```
SDK Remote -> Relay -> TunnelClient -> SyscallRouter -> Module -> TunnelClient -> Relay -> SDK Remote
```

### LLM Flow (SDK-local)
```
CloveClient.think() -> llm_service.py subprocess -> Gemini API -> response
```

## 10) Architectural Recommendations

1) **Replace shell-based exec/http**
   - `SYS_EXEC` and `SYS_HTTP` use `popen` + string concatenation.
   - Move to execve/argv + a proper HTTP client to reduce injection surface and stderr/stdout confusion.

2) **Make async execution more uniform**
   - Any long-running syscall should be async-first to avoid reactor stalls.
   - Consider a per-syscall async path or a unified async adapter layer.

3) **LLM accounting and quotas**
   - Kernel no longer executes LLM, but still stores LLM quotas/usage.
   - Decide: either remove LLM quota fields from kernel or let SDK report usage back explicitly.

4) **LLM service lifecycle**
   - Spawn-per-call LLM is slow and resource heavy.
   - Move to a long-lived local LLM service process or HTTP microservice for reuse.

5) **Protocol versioning**
   - Add a version field in the header or a handshake syscall to support forward compatibility.

6) **Security tightening**
   - Consider default deny for `SYS_EXEC`/`SYS_HTTP` unless explicitly allowed.
   - Extend permissions to include explicit allowlists for binaries and HTTP methods.

7) **Observability alignment**
   - LLM metrics now live outside kernel; adjust dashboards to avoid stale or misleading LLM counters.

