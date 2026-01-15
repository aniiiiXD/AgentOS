# AgentOS

A microkernel for AI agents. Run multiple AI agents as isolated processes with shared LLM access.

> **"systemd for AI agents"** | **"Docker-lite for autonomous compute"**

## Why AgentOS?

**The Problem:** Python agent frameworks (LangChain, CrewAI, AutoGen) run agents as coroutines or threads. When one agent crashes, leaks memory, or infinite loops - it takes down the entire system.

**The Solution:** AgentOS provides **OS-level isolation**. Each agent is a real process with:
- **Fault isolation** - One agent crashes, others continue
- **Resource limits** - Memory/CPU caps enforced by the kernel (cgroups)
- **Security sandboxing** - Untrusted agents can't access filesystem/network
- **Fair scheduling** - Shared LLM access with rate limiting

**This is literally why operating systems exist** - and AgentOS brings these guarantees to AI agents.

### What AgentOS Can Do That Python Frameworks Can't

| Scenario | Python Frameworks | AgentOS |
|----------|-------------------|---------|
| Agent infinite loops | Entire system hangs | Agent throttled, others continue |
| Agent memory leak | OOM kills everything | Only that agent killed |
| Malicious agent code | Full system access | Sandboxed, access denied |
| 10 agents need LLM | Race conditions | Fair queuing & scheduling |
| Agent crashes | May corrupt shared state | Clean isolation |

## What is AgentOS?

AgentOS treats AI agents like operating system processes. It provides:

- **Process Isolation**: Each agent runs in a sandboxed environment (Linux namespaces + cgroups)
- **Shared LLM Access**: All agents share a single LLM connection (Gemini API)
- **IPC Protocol**: Binary protocol over Unix domain sockets for fast communication
- **Resource Limits**: Control memory, CPU, and process limits per agent

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentOS Kernel (C++)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Reactor   │  │  LLM Client │  │   Agent Manager     │  │
│  │   (epoll)   │  │   (Gemini)  │  │   (Sandbox/Fork)    │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         └────────────────┼─────────────────────┘             │
│                          │                                   │
│              Unix Domain Socket (/tmp/agentos.sock)          │
└──────────────────────────┼───────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────┴────┐        ┌────┴────┐        ┌────┴────┐
   │ Agent 1 │        │ Agent 2 │        │ Agent 3 │
   │(Python) │        │(Python) │        │(Python) │
   └─────────┘        └─────────┘        └─────────┘
```

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Unix Socket IPC | Done | Binary protocol with 17-byte header |
| Event Loop | Done | epoll-based reactor pattern |
| LLM Integration | Done | Google Gemini API (gemini-2.0-flash) |
| Process Sandboxing | Done | Linux namespaces (PID, NET, MNT, UTS) |
| Resource Limits | Done | cgroups v2 (memory, CPU, PIDs) |
| Python SDK | Done | Full client library |

## Requirements

- Linux (Ubuntu 22.04+ / Debian 12+)
- GCC 11+ with C++23 support
- CMake 3.20+
- Python 3.10+
- vcpkg (package manager)

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url> AgentOS
cd AgentOS

# Install vcpkg if you don't have it
git clone https://github.com/Microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT="$HOME/vcpkg"
```

### 2. Install Dependencies

```bash
sudo apt install -y build-essential cmake pkg-config libssl-dev python3

# vcpkg will install C++ dependencies automatically during build
```

### 3. Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 4. Run the Kernel

```bash
# Without LLM (for testing)
./agentos_kernel

# With Gemini LLM
export GEMINI_API_KEY="your-api-key"
./agentos_kernel
```

### 5. Run an Agent

In another terminal:

```bash
# Simple echo test
python3 agents/examples/hello_agent.py

# LLM interaction (requires API key)
python3 agents/examples/thinking_agent.py

# Spawn/kill agents test
python3 agents/examples/spawn_test.py
```

## OS-Level Demonstrations

These demos prove why AgentOS must exist - they show capabilities that **only** an OS-level kernel can provide.

### Fault Isolation Demo

Spawns 3 agents: a CPU hog, a memory leaker, and a healthy agent. Proves that misbehaving agents don't crash the system.

```bash
# User mode (limited isolation)
python3 agents/examples/fault_isolation_demo.py

# Root mode (full cgroups enforcement)
sudo ./build/agentos_kernel &
sudo python3 agents/examples/fault_isolation_demo.py
```

**What happens:**
```
Agent         Status        Notes
────────────────────────────────────────
cpu-hog       THROTTLED     CPU limited to 10%
mem-hog       KILLED        OOM at 50MB limit
healthy       RUNNING       Unaffected, still working
```

**Why this matters:** This is literally why operating systems exist. Python frameworks can't do this.

### Individual Stress Test Agents

```bash
# CPU stress test (infinite loop - gets throttled)
python3 agents/examples/cpu_hog_agent.py

# Memory stress test (allocates until killed)
python3 agents/examples/memory_hog_agent.py

# Well-behaved agent (survives while others fail)
python3 agents/examples/healthy_agent.py
```

## Python SDK Usage

```python
import sys
sys.path.insert(0, 'agents/python_sdk')
from agentos import AgentOSClient

with AgentOSClient() as client:
    # Echo test
    response = client.echo("Hello!")
    print(response)  # "Hello!"

    # Ask the LLM
    result = client.think("What is 2+2?")
    print(result['content'])  # "4"

    # Spawn a new agent
    agent = client.spawn(
        name="worker1",
        script="/path/to/worker.py",
        sandboxed=True,
        limits={"memory": 256*1024*1024}
    )
    print(f"Spawned: {agent}")

    # List running agents
    agents = client.list_agents()
    print(agents)

    # Kill an agent
    client.kill(name="worker1")
```

## Syscall Reference

| Opcode | Name | Description | Payload | Response |
|--------|------|-------------|---------|----------|
| 0x00 | SYS_NOOP | Echo test | Any bytes | Same bytes |
| 0x01 | SYS_THINK | LLM query | Prompt string | JSON: `{content, tokens, error}` |
| 0x10 | SYS_SPAWN | Spawn agent | JSON config | JSON: `{id, name, pid, status}` |
| 0x11 | SYS_KILL | Kill agent | JSON: `{name}` or `{id}` | JSON: `{killed: bool}` |
| 0x12 | SYS_LIST | List agents | Empty | JSON array of agents |
| 0xFF | SYS_EXIT | Disconnect | Empty | "goodbye" |

## Example Workflows

### 1. Research Assistant

An agent that searches the web and summarizes findings:

```python
# agents/examples/research_agent.py
import sys
sys.path.insert(0, '../python_sdk')
from agentos import AgentOSClient

def research(topic):
    with AgentOSClient() as client:
        # Ask LLM to generate search queries
        result = client.think(f"""
        I need to research: {topic}
        Generate 3 specific search queries to find information about this.
        Format: one query per line.
        """)

        queries = result['content'].strip().split('\n')

        # For each query, ask LLM to analyze (in real app, you'd fetch URLs)
        findings = []
        for query in queries[:3]:
            analysis = client.think(f"""
            Search query: {query}
            Based on your knowledge, provide key facts about this topic.
            Be concise (2-3 sentences).
            """)
            findings.append(analysis['content'])

        # Synthesize findings
        summary = client.think(f"""
        Topic: {topic}

        Findings:
        {chr(10).join(f'- {f}' for f in findings)}

        Write a brief summary paragraph combining these findings.
        """)

        return summary['content']

if __name__ == '__main__':
    topic = input("Research topic: ")
    result = research(topic)
    print(f"\n=== Summary ===\n{result}")
```

### 2. Code Review Agent

An agent that reviews code and suggests improvements:

```python
# agents/examples/code_review_agent.py
import sys
sys.path.insert(0, '../python_sdk')
from agentos import AgentOSClient

def review_code(code: str, language: str = "python"):
    with AgentOSClient() as client:
        result = client.think(f"""
        Review this {language} code for:
        1. Bugs or errors
        2. Security issues
        3. Performance problems
        4. Style improvements

        Code:
        ```{language}
        {code}
        ```

        Provide specific, actionable feedback.
        """)
        return result['content']

if __name__ == '__main__':
    code = '''
def get_user(id):
    query = "SELECT * FROM users WHERE id = " + id
    return db.execute(query)
    '''

    print(review_code(code))
```

### 3. Multi-Agent Chat System

Spawn multiple specialized agents that collaborate:

```python
# agents/examples/multi_agent_chat.py
import sys
import time
sys.path.insert(0, '../python_sdk')
from agentos import AgentOSClient

def create_specialist_agent(client, name, role):
    """Create a worker agent with a specific role."""
    # In a full implementation, each worker would have its own context
    return client.spawn(
        name=name,
        script="worker_agent.py",
        sandboxed=True,
        limits={"memory": 128*1024*1024, "max_pids": 16}
    )

def multi_agent_discussion(topic):
    with AgentOSClient() as client:
        # Different perspectives
        perspectives = [
            ("analyst", "analytical and data-driven"),
            ("creative", "creative and unconventional"),
            ("critic", "critical and skeptical"),
        ]

        responses = []
        for name, style in perspectives:
            result = client.think(f"""
            You are a {style} thinker.
            Topic: {topic}

            Share your perspective in 2-3 sentences.
            Be {style} in your approach.
            """)
            responses.append((name, result['content']))
            print(f"\n[{name.upper()}]: {result['content']}")

        # Synthesize
        all_perspectives = "\n".join(f"- {name}: {resp}" for name, resp in responses)
        synthesis = client.think(f"""
        Topic: {topic}

        Different perspectives:
        {all_perspectives}

        Synthesize these viewpoints into a balanced conclusion (2-3 sentences).
        """)

        print(f"\n[SYNTHESIS]: {synthesis['content']}")
        return synthesis['content']

if __name__ == '__main__':
    topic = input("Discussion topic: ")
    multi_agent_discussion(topic)
```

### 4. Autonomous Task Agent

An agent that breaks down and executes complex tasks:

```python
# agents/examples/task_agent.py
import sys
sys.path.insert(0, '../python_sdk')
from agentos import AgentOSClient

def execute_task(task_description):
    with AgentOSClient() as client:
        # Plan
        plan = client.think(f"""
        Task: {task_description}

        Break this into 3-5 concrete steps.
        Format each step as: "STEP N: description"
        """)
        print(f"=== Plan ===\n{plan['content']}\n")

        # Extract steps
        steps = [line.strip() for line in plan['content'].split('\n')
                 if line.strip().startswith('STEP')]

        # Execute each step
        results = []
        for i, step in enumerate(steps, 1):
            print(f"\n--- Executing {step} ---")

            result = client.think(f"""
            You are executing step {i} of a task.

            Overall task: {task_description}
            Current step: {step}
            Previous results: {results[-1] if results else 'None'}

            Provide the output for this step.
            """)

            results.append(result['content'])
            print(f"Result: {result['content'][:200]}...")

        # Final summary
        summary = client.think(f"""
        Task completed: {task_description}

        Step results:
        {chr(10).join(f'{i+1}. {r[:100]}...' for i, r in enumerate(results))}

        Provide a brief final summary of what was accomplished.
        """)

        print(f"\n=== Final Summary ===\n{summary['content']}")
        return summary['content']

if __name__ == '__main__':
    task = input("Task: ")
    execute_task(task)
```

### 5. Conversation Memory Agent

An agent that maintains conversation context:

```python
# agents/examples/memory_agent.py
import sys
sys.path.insert(0, '../python_sdk')
from agentos import AgentOSClient

class ConversationAgent:
    def __init__(self):
        self.history = []
        self.client = AgentOSClient()
        self.client.connect()

    def chat(self, user_message):
        self.history.append(f"User: {user_message}")

        # Build context from history
        context = "\n".join(self.history[-10:])  # Last 10 messages

        result = self.client.think(f"""
        Conversation history:
        {context}

        Respond naturally to the user's latest message.
        Keep your response concise (1-3 sentences).
        """)

        assistant_response = result['content']
        self.history.append(f"Assistant: {assistant_response}")

        return assistant_response

    def close(self):
        self.client.disconnect()

if __name__ == '__main__':
    agent = ConversationAgent()
    print("Chat with the agent (type 'quit' to exit)\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'quit':
                break
            if not user_input:
                continue

            response = agent.chat(user_input)
            print(f"Agent: {response}\n")
    finally:
        agent.close()
```

## Configuration

### Kernel Configuration

The kernel can be configured programmatically:

```cpp
// In your code
agentos::kernel::KernelConfig config;
config.socket_path = "/tmp/agentos.sock";
config.enable_sandboxing = true;
config.gemini_api_key = "your-key";  // Or set GEMINI_API_KEY env var
config.llm_model = "gemini-2.0-flash";

agentos::kernel::Kernel kernel(config);
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | (none) |
| `GOOGLE_API_KEY` | Fallback API key | (none) |

### Agent Resource Limits

When spawning agents, you can set resource limits:

```python
client.spawn(
    name="worker",
    script="/path/to/script.py",
    sandboxed=True,
    limits={
        "memory": 256 * 1024 * 1024,  # 256MB
        "max_pids": 64,                # Max processes
        "cpu_quota": 100000            # CPU microseconds per period
    }
)
```

## Project Structure

```
AgentOS/
├── src/
│   ├── main.cpp                 # Entry point
│   ├── kernel/
│   │   ├── kernel.hpp/cpp       # Main kernel class
│   │   ├── reactor.hpp/cpp      # epoll event loop
│   │   └── llm_client.hpp/cpp   # Gemini API client
│   ├── ipc/
│   │   ├── protocol.hpp         # Binary protocol definition
│   │   └── socket_server.hpp/cpp # Unix socket server
│   ├── runtime/
│   │   ├── sandbox.hpp/cpp      # Linux namespaces/cgroups
│   │   └── agent_process.hpp/cpp # Agent lifecycle
│   └── util/
│       └── logger.hpp/cpp       # Logging utilities
├── agents/
│   ├── python_sdk/
│   │   └── agentos.py           # Python client library
│   └── examples/
│       ├── hello_agent.py       # Echo test
│       ├── thinking_agent.py    # LLM interaction
│       ├── worker_agent.py      # Spawnable worker
│       ├── spawn_test.py        # Agent management test
│       ├── fault_isolation_demo.py  # OS-level fault isolation demo
│       ├── cpu_hog_agent.py     # CPU stress test agent
│       ├── memory_hog_agent.py  # Memory stress test agent
│       └── healthy_agent.py     # Well-behaved agent
├── build/                       # Build output
├── CMakeLists.txt              # Build configuration
├── vcpkg.json                  # C++ dependencies
├── STATUS.md                   # Development status
└── README.md                   # This file
```

## Troubleshooting

### "Permission denied" on socket

```bash
rm -f /tmp/agentos.sock
./agentos_kernel
```

### "LLM not configured" error

```bash
export GEMINI_API_KEY="your-api-key"
./agentos_kernel
```

### Sandbox requires root

Full namespace isolation requires root privileges:

```bash
sudo ./agentos_kernel
```

Without root, agents run with fork() instead of clone() with namespaces.

### Build errors

```bash
# Clean rebuild
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Development

### Adding a New Syscall

1. Add opcode to `src/ipc/protocol.hpp`
2. Add handler method in `src/kernel/kernel.hpp`
3. Implement handler in `src/kernel/kernel.cpp`
4. Update Python SDK in `agents/python_sdk/agentos.py`

### Running Tests

```bash
# Start kernel
./build/agentos_kernel &

# Run basic tests
python3 agents/examples/hello_agent.py
python3 agents/examples/spawn_test.py
python3 agents/examples/thinking_agent.py

# Run OS-level fault isolation demo
python3 agents/examples/fault_isolation_demo.py

# Stop kernel
pkill agentos_kernel
```

### Running with Full Isolation (requires root)

```bash
# Start kernel with cgroups enabled
sudo ./build/agentos_kernel &

# Run fault isolation demo (will enforce memory/CPU limits)
sudo python3 agents/examples/fault_isolation_demo.py

# Stop kernel
sudo pkill agentos_kernel
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

Built with C++23, powered by Google Gemini.
