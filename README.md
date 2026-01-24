# Clove

A microkernel runtime for AI agents. Like Postgres for databases — Clove is the server that provides OS-level isolation, resource limits, and sandboxing for your agents.

```
┌─────────────────────────────────┐
│  Your Agent Code (Python)       │  ← pip install clove-sdk
│  from clove_sdk import ...      │
└────────────┬────────────────────┘
             │ connects to
┌────────────▼────────────────────┐
│  Clove Runtime                  │  ← docker / binary / cloud
│  • Process isolation            │
│  • cgroups resource limits      │
│  • Linux namespace sandboxing   │
│  • LLM access (Gemini)         │
│  • Inter-agent IPC             │
└─────────────────────────────────┘
```

---

## Start the Runtime

**Docker** (recommended):
```bash
docker run -d --privileged -e GEMINI_API_KEY=xxx ghcr.io/anixd/clove
```

**Local binary:**
```bash
curl -fsSL https://raw.githubusercontent.com/anixd/clove/main/install.sh | bash
export GEMINI_API_KEY=xxx
clove_kernel
```

**From source:**
```bash
git clone https://github.com/anixd/clove.git && cd clove
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
GEMINI_API_KEY=xxx ./clove_kernel
```

---

## Install the SDK

```bash
pip install clove-sdk
```

---

## Write an Agent

```python
from clove_sdk import CloveClient

with CloveClient() as client:
    # Query LLM
    result = client.think("Summarize the theory of relativity in one sentence")
    print(result['content'])

    # Execute commands (sandboxed)
    output = client.exec("ls /tmp")
    print(output['stdout'])

    # Spawn a child agent with resource limits
    client.spawn(
        name="worker",
        script="/path/to/worker.py",
        sandboxed=True,
        limits={"memory": 256 * 1024 * 1024, "cpu_quota": 50000}
    )

    # Check what's running
    agents = client.list_agents()
    print(agents)
```

---

## Why Clove?

Python frameworks (LangChain, CrewAI, AutoGen) run agents as threads/coroutines. One bad agent takes down everything.

Clove runs agents as **real OS processes** with:

| | Python Frameworks | Clove |
|--|-------------------|-------|
| Agent infinite loops | System hangs | Agent throttled |
| Memory leak | OOM kills all | Only that agent killed |
| Malicious code | Full access | Sandboxed |
| 10 agents need LLM | Race conditions | Fair queuing |
| Agent crash | Corrupts shared state | Clean isolation |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    Clove Kernel (C++23)                          │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │   Reactor   │  │   LLM Client    │  │   Agent Manager     │ │
│  │   (epoll)   │  │ (subprocess)    │  │   (Sandbox/Fork)    │ │
│  └──────┬──────┘  └────────┬────────┘  └──────────┬──────────┘ │
│         └──────────────────┼───────────────────────┘            │
│                            │                                    │
│              Unix Domain Socket (/tmp/clove.sock)               │
└────────────────────────────┼────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
   │ Agent 1 │          │ Agent 2 │          │ Agent 3 │
   └─────────┘          └─────────┘          └─────────┘
```

- **Kernel** — C++23 event loop, manages agent lifecycle
- **Agents** — Python processes, communicate via syscalls
- **SDK** — `pip install clove-sdk`, connects to the runtime

---

## Features

| Feature | Description |
|---------|-------------|
| Process isolation | Linux namespaces (PID, NET, MNT, UTS) |
| Resource limits | cgroups v2 (memory, CPU, PIDs) |
| LLM integration | Gemini API, fair queuing across agents |
| Permission system | Path validation, command filtering, domain whitelist |
| Hot reload | Auto-restart crashed agents with backoff |
| IPC | Inter-agent messaging + pub/sub events |
| Metrics | Per-agent CPU, memory, syscall counts |
| World simulation | Virtual filesystems, network mocking, chaos injection |
| Remote access | Relay server for cloud deployments |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | For LLM | Google Gemini API key |

---

## Docker Modes

```bash
# Run as a service (default)
docker run -d --privileged -e GEMINI_API_KEY=xxx ghcr.io/anixd/clove

# Interactive shell with kernel running
docker run --rm -it --privileged -e GEMINI_API_KEY=xxx ghcr.io/anixd/clove shell

# Run the demo
docker run --rm -it --privileged ghcr.io/anixd/clove demo
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Syscalls Reference](docs/syscalls.md) | All available syscalls |
| [World Simulation](docs/world-simulation.md) | Virtual worlds for agent testing |
| [Hot Reload](docs/hot-reload.md) | Auto-restart and recovery |
| [Python SDK](agents/python_sdk/README.md) | Full SDK API reference |

---

## Requirements

- **Runtime:** Linux x86_64 (Ubuntu 22.04+), root for full isolation
- **SDK:** Python 3.10+, any platform (connects to runtime)

## License

MIT
