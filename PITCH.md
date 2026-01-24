# Clove: Pitch & Application Guide

> How to pitch Clove for jobs, internships, or investor conversations.

---

## The One-Liner

> "Everyone's building AI agents. Nobody's solving where they run. I built that."

---

## What Clove Demonstrates

| Skill | Evidence |
|-------|----------|
| **Systems Programming** | C++ kernel, epoll reactor, Unix domain sockets, binary protocol |
| **OS Internals** | Linux namespaces, cgroups v2, process isolation, sandboxing |
| **Architecture Design** | Kernel ↔ SDK separation, IPC protocol, permission system |
| **Full-Stack Thinking** | C++ kernel + Python SDK + CLI + Web dashboard |
| **Shipping** | Working system with tests, documentation, multiple components |
| **Problem Identification** | Spotted a real gap in the AI tooling ecosystem |

This isn't a tutorial project. It's production infrastructure.

---

## The Pitch (Short - DM/Email)

> Building agents is easy now. Running them isn't.
>
> I built Clove - a microkernel for AI agents. Each agent = real Linux process. One crashes, others continue, auto-restart with backoff.
>
> OS-level isolation for AI workloads.
>
> 60-sec demo: [link]
> GitHub: [link]

---

## The Pitch (Medium - Application)

> **The Problem I Noticed**
>
> Everyone's building AI agents. Vibe coding makes it trivial to create an agent swarm in 30 minutes. But where does it run?
>
> Python frameworks (LangChain, CrewAI) run agents as coroutines. One infinite loop freezes everything. One memory leak OOMs the whole system. One crash corrupts shared state.
>
> This is literally why operating systems exist - and nobody's applied those lessons to AI agents.
>
> **What I Built**
>
> Clove is a microkernel for AI agents. Written in C++, it provides:
>
> - **Process isolation**: Each agent is a real Linux process
> - **Crash recovery**: One dies, others continue, auto-restart with backoff
> - **Resource limits**: cgroups prevent memory/CPU runaway
> - **Sandboxing**: Linux namespaces for untrusted agents
> - **IPC**: Agents communicate via kernel-mediated messaging
>
> It's systemd for AI agents.
>
> **Links**
> - Demo: [link]
> - GitHub: [link]

---

## The Pitch (Long - Interview/Deep Dive)

### The Problem

Everyone's building AI agents. The tooling for *creating* agents is great - LangChain, CrewAI, AutoGen, vibe coding. But the tooling for *running* agents at scale doesn't exist.

Python frameworks run agents as coroutines or threads:
- One infinite loop → entire system hangs
- One memory leak → OOM kills everything
- One crash → corrupts shared state
- No fair scheduling for shared LLM access

This is literally why operating systems were invented. Process isolation, resource limits, fair scheduling - these are solved problems. Nobody's applied them to AI agents.

### The Solution

Clove is a microkernel that runs AI agents as isolated processes.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Clove Kernel (C++)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Reactor   │  │  LLM Client │  │   Agent Manager     │  │
│  │   (epoll)   │  │ (subprocess)│  │   (Sandbox/Fork)    │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         └────────────────┼────────────────────┘             │
│                          │                                   │
│              Unix Domain Socket (/tmp/clove.sock)            │
└──────────────────────────┼───────────────────────────────────┘
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
 ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
 │ Agent 1 │          │ Agent 2 │          │ Agent 3 │
 │(Python) │          │(Python) │          │(Python) │
 └─────────┘          └─────────┘          └─────────┘
```

**Key technical decisions:**

1. **Binary protocol over Unix sockets** - 17-byte header (magic + agent_id + opcode + payload_size) + JSON payload. Low overhead, type-safe at the wire level.

2. **epoll reactor** - Single-threaded event loop handles all I/O. Scales to thousands of connections without thread overhead.

3. **Linux namespaces for sandboxing** - PID, NET, MNT, UTS namespaces isolate untrusted agents. They can't see other processes or access the network.

4. **cgroups v2 for resource limits** - Memory limits prevent OOM cascades. CPU quotas ensure fair scheduling. PID limits prevent fork bombs.

5. **Hot reload with exponential backoff** - Crashed agents auto-restart. Backoff prevents restart loops. Escalation after max retries.

### What's Implemented

- Core kernel (C++, ~90k lines across all components)
- Python SDK with full syscall coverage
- Permission system (path validation, command filtering, domain whitelist)
- Inter-agent IPC (send, receive, broadcast)
- Shared state store (key-value with TTL and scopes)
- Event system (pub/sub for kernel events)
- Remote connectivity (WebSocket tunnel to relay server)
- World Engine (simulated environments for testing)
- CLI tool for deployment and management
- Metrics collection (CPU, memory, disk, network per agent)

### What I Learned

1. **Protocol design matters** - The binary header makes parsing deterministic. JSON payloads keep it flexible. This split works well.

2. **Sandboxing is hard** - Linux namespaces have subtle gotchas. You need to set up /proc correctly, handle signal propagation, manage cleanup on crash.

3. **Hot reload needs thought** - Naive restart causes loops. Exponential backoff + max retries + escalation callbacks handle it gracefully.

4. **The real value is in the ops story** - The kernel is table stakes. What matters is: deploy to any cloud, monitor everything, auto-recover from failures.

---

## Framing for Different Audiences

### AI Infrastructure Company
> "You make building agents easy. I'm solving where they run. Clove is the infrastructure layer for the agents your users create."

### Systems/Infrastructure Role
> "I built a microkernel in C++ with epoll, Unix sockets, Linux namespaces, and cgroups. It runs AI agents as isolated processes with auto-recovery."

### Startup/Generalist Role
> "I noticed a gap in the AI tooling ecosystem - everyone's building agents, nobody's solving deployment. I built the infrastructure layer."

### Research/Academic
> "I'm exploring OS-level primitives for AI agent isolation. The kernel provides process isolation, resource limits, and IPC for multi-agent systems."

---

## Before Applying: Checklist

- [ ] **Demo video** (60 seconds showing crash isolation)
- [ ] **README hook** (why should I care, not just how it works)
- [ ] **One-liner install** (`curl ... | sh` or docker-compose)
- [ ] **Benchmarks** (crash recovery time, IPC overhead, memory isolation)
- [ ] **Social proof** (5 people tried it, here's what they said)

---

## Sample Tweets for Launch

**The Hook:**
> You vibe-coded an agent swarm in 30 minutes.
>
> Cool. Where does it run?
>
> Your laptop can't handle 50 agents. Lambda times out. Kubernetes is overkill.
>
> You need agent infrastructure.

**The Problem:**
> LangChain agents are just Python coroutines.
>
> One infinite loop = entire system frozen.
> One memory leak = OOM kills everything.
> One crash = corrupted shared state.
>
> This is literally why operating systems exist.

**The Solution:**
> Clove: systemd for AI agents.
>
> - Each agent = real Linux process
> - One crashes → others continue
> - Memory leak → only that agent killed
> - Auto-restart with backoff
>
> OS-level isolation for AI workloads.

**The Demo:**
> Spawned 100 agents on Clove.
> Killed 20 randomly.
> 80 continued working.
> 20 auto-restarted.
>
> Try that with CrewAI.

---

## Remember

The difference between "good project" and "gets the interview" is **packaging**.

- Demo video > wall of text
- "Here's the problem I noticed" > "Here's what I built"
- Show you understand THEIR problem > generic application
- Numbers and proof > claims

---

*Last updated: 2025-01-22*
