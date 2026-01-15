#include "kernel/kernel.hpp"
#include <spdlog/spdlog.h>
#include <sys/epoll.h>
#include <csignal>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace agentos::kernel {

// Global kernel pointer for signal handling
static Kernel* g_kernel = nullptr;

static void signal_handler(int signum) {
    spdlog::info("Received signal {}, shutting down...", signum);
    if (g_kernel) {
        g_kernel->shutdown();
    }
}

Kernel::Kernel()
    : Kernel(Config{}) {}

Kernel::Kernel(const Config& config)
    : config_(config)
    , reactor_(std::make_unique<Reactor>())
    , socket_server_(std::make_unique<ipc::SocketServer>(config.socket_path))
    , agent_manager_(std::make_unique<runtime::AgentManager>(config.socket_path))
{
    // Initialize LLM client
    LLMConfig llm_config;
    llm_config.api_key = config.gemini_api_key;
    llm_config.model = config.llm_model;
    llm_client_ = std::make_unique<LLMClient>(llm_config);
}

Kernel::~Kernel() {
    if (g_kernel == this) {
        g_kernel = nullptr;
    }
}

bool Kernel::init() {
    spdlog::info("Initializing AgentOS Kernel...");

    // Initialize reactor
    if (!reactor_->init()) {
        spdlog::error("Failed to initialize reactor");
        return false;
    }

    // Set up message handler
    socket_server_->set_handler([this](const ipc::Message& msg) {
        return handle_message(msg);
    });

    // Initialize socket server
    if (!socket_server_->init()) {
        spdlog::error("Failed to initialize socket server");
        return false;
    }

    // Add server socket to reactor
    int server_fd = socket_server_->get_server_fd();
    reactor_->add(server_fd, EPOLLIN, [this](int fd, uint32_t events) {
        on_server_event(fd, events);
    });

    // Set up signal handlers
    g_kernel = this;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    spdlog::info("Kernel initialized successfully");
    spdlog::info("Sandboxing: {}", config_.enable_sandboxing ? "enabled" : "disabled");
    spdlog::info("LLM: {} ({})",
        llm_client_->is_configured() ? "configured" : "not configured",
        llm_client_->config().model);
    return true;
}

void Kernel::run() {
    running_ = true;
    spdlog::info("AgentOS Kernel v0.1.0 running");
    spdlog::info("Listening on: {}", config_.socket_path);
    spdlog::info("Press Ctrl+C to exit");

    while (running_) {
        int n = reactor_->poll(100);
        if (n < 0) {
            spdlog::error("Reactor error, exiting");
            break;
        }

        // Reap dead agents periodically
        agent_manager_->reap_agents();
    }

    spdlog::info("Kernel shutting down...");
    agent_manager_->stop_all();
    socket_server_->stop();
    spdlog::info("Kernel stopped");
}

void Kernel::shutdown() {
    running_ = false;
}

void Kernel::on_server_event(int fd, uint32_t events) {
    if (events & EPOLLIN) {
        // Accept new connections
        while (true) {
            int client_fd = socket_server_->accept_connection();
            if (client_fd < 0) {
                break;
            }

            // Add client to reactor
            reactor_->add(client_fd, EPOLLIN | EPOLLHUP | EPOLLERR,
                [this](int cfd, uint32_t ev) {
                    on_client_event(cfd, ev);
                });
        }
    }
}

void Kernel::on_client_event(int fd, uint32_t events) {
    // Handle errors and hangups
    if (events & (EPOLLHUP | EPOLLERR)) {
        reactor_->remove(fd);
        socket_server_->remove_client(fd);
        return;
    }

    // Handle readable
    if (events & EPOLLIN) {
        if (!socket_server_->handle_client(fd)) {
            reactor_->remove(fd);
            socket_server_->remove_client(fd);
            return;
        }
    }

    // Handle writable
    if (events & EPOLLOUT) {
        if (!socket_server_->flush_client(fd)) {
            reactor_->remove(fd);
            socket_server_->remove_client(fd);
            return;
        }
    }

    // Update events based on write buffer
    update_client_events(fd);
}

void Kernel::update_client_events(int fd) {
    uint32_t events = EPOLLIN | EPOLLHUP | EPOLLERR;
    if (socket_server_->client_wants_write(fd)) {
        events |= EPOLLOUT;
    }
    reactor_->modify(fd, events);
}

ipc::Message Kernel::handle_message(const ipc::Message& msg) {
    switch (msg.opcode) {
        case ipc::SyscallOp::SYS_NOOP:
            // Echo the payload back
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_NOOP, msg.payload);

        case ipc::SyscallOp::SYS_EXIT:
            spdlog::info("Agent {} requested exit", msg.agent_id);
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EXIT, "goodbye");

        case ipc::SyscallOp::SYS_THINK:
            return handle_think(msg);

        case ipc::SyscallOp::SYS_SPAWN:
            return handle_spawn(msg);

        case ipc::SyscallOp::SYS_KILL:
            return handle_kill(msg);

        case ipc::SyscallOp::SYS_LIST:
            return handle_list(msg);

        default:
            spdlog::warn("Unknown opcode: 0x{:02x}", static_cast<uint8_t>(msg.opcode));
            return ipc::Message(msg.agent_id, msg.opcode, msg.payload);
    }
}

ipc::Message Kernel::handle_think(const ipc::Message& msg) {
    // Check if LLM is configured
    if (!llm_client_->is_configured()) {
        json response;
        response["error"] = "LLM not configured (set GEMINI_API_KEY)";
        response["content"] = "";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_THINK, response.dump());
    }

    std::string prompt = msg.payload_str();
    spdlog::debug("Agent {} thinking: {}", msg.agent_id,
        prompt.length() > 50 ? prompt.substr(0, 50) + "..." : prompt);

    // Call Gemini API
    auto result = llm_client_->complete(prompt);

    json response;
    if (result.success) {
        response["content"] = result.content;
        response["tokens"] = result.tokens_used;
        spdlog::debug("LLM response: {} tokens", result.tokens_used);
    } else {
        response["error"] = result.error;
        response["content"] = "";
        spdlog::warn("LLM error for agent {}: {}", msg.agent_id, result.error);
    }

    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_THINK, response.dump());
}

ipc::Message Kernel::handle_spawn(const ipc::Message& msg) {
    // Parse JSON payload: {"name": "agent1", "script": "/path/to/script.py"}
    try {
        json j = json::parse(msg.payload_str());

        runtime::AgentConfig config;
        config.name = j.value("name", "agent_" + std::to_string(runtime::AgentProcess::generate_id()));
        config.script_path = j.value("script", "");
        config.python_path = j.value("python", "python3");
        config.sandboxed = config_.enable_sandboxing && j.value("sandboxed", true);
        config.enable_network = j.value("network", false);

        // Resource limits
        if (j.contains("limits")) {
            auto& lim = j["limits"];
            config.limits.memory_limit_bytes = lim.value("memory", 256 * 1024 * 1024);
            config.limits.max_pids = lim.value("max_pids", 64);
            config.limits.cpu_quota_us = lim.value("cpu_quota", 100000);
        }

        if (config.script_path.empty()) {
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SPAWN,
                R"({"error": "script path required"})");
        }

        spdlog::info("Spawning agent: {} (script={})", config.name, config.script_path);

        auto agent = agent_manager_->spawn_agent(config);
        if (!agent) {
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SPAWN,
                R"({"error": "failed to spawn agent"})");
        }

        json response;
        response["id"] = agent->id();
        response["name"] = agent->name();
        response["pid"] = agent->pid();
        response["status"] = "running";

        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SPAWN, response.dump());

    } catch (const std::exception& e) {
        spdlog::error("Failed to parse spawn request: {}", e.what());
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SPAWN,
            R"({"error": "invalid JSON"})");
    }
}

ipc::Message Kernel::handle_kill(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        bool killed = false;
        if (j.contains("id")) {
            killed = agent_manager_->kill_agent(j["id"].get<uint32_t>());
        } else if (j.contains("name")) {
            killed = agent_manager_->kill_agent(j["name"].get<std::string>());
        }

        json response;
        response["killed"] = killed;

        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_KILL, response.dump());

    } catch (const std::exception& e) {
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_KILL,
            R"({"error": "invalid request"})");
    }
}

ipc::Message Kernel::handle_list(const ipc::Message& msg) {
    json response = json::array();

    for (const auto& agent : agent_manager_->list_agents()) {
        json a;
        a["id"] = agent->id();
        a["name"] = agent->name();
        a["pid"] = agent->pid();
        a["state"] = runtime::agent_state_to_string(agent->state());
        a["running"] = agent->is_running();
        response.push_back(a);
    }

    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_LIST, response.dump());
}

} // namespace agentos::kernel
