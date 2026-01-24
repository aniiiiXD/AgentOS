#include "kernel/kernel.hpp"
#include <spdlog/spdlog.h>
#include <sys/epoll.h>
#include <sys/wait.h>
#include <csignal>
#include <fstream>
#include <thread>
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
    , world_engine_(std::make_unique<WorldEngine>())
    , tunnel_client_(std::make_unique<TunnelClient>())
    , metrics_collector_(std::make_unique<clove::metrics::MetricsCollector>())
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

    // Set up restart event callback for auto-recovery notifications
    agent_manager_->set_restart_event_callback(
        [this](const std::string& event_type, const std::string& agent_name,
               uint32_t restart_count, int exit_code) {
            json event_data;
            event_data["agent_name"] = agent_name;
            event_data["restart_count"] = restart_count;
            event_data["exit_code"] = exit_code;

            if (event_type == "AGENT_RESTARTING") {
                emit_event(KernelEventType::AGENT_RESTARTING, event_data, 0);
            } else if (event_type == "AGENT_ESCALATED") {
                emit_event(KernelEventType::AGENT_ESCALATED, event_data, 0);
            }
        });

    // Initialize tunnel client
    if (tunnel_client_->init()) {
        spdlog::info("Tunnel client initialized");

        // Configure if relay URL is set
        if (!config_.relay_url.empty()) {
            TunnelConfig tc;
            tc.relay_url = config_.relay_url;
            tc.machine_id = config_.machine_id;
            tc.token = config_.machine_token;
            tunnel_client_->configure(tc);

            // Auto-connect if configured
            if (config_.tunnel_auto_connect) {
                if (tunnel_client_->connect()) {
                    spdlog::info("Tunnel connected to relay: {}", config_.relay_url);
                } else {
                    spdlog::warn("Failed to auto-connect tunnel to relay");
                }
            }
        }
    }

    spdlog::info("Kernel initialized successfully");
    spdlog::info("Sandboxing: {}", config_.enable_sandboxing ? "enabled" : "disabled");
    spdlog::info("LLM: {} ({})",
        llm_client_->is_configured() ? "configured" : "not configured",
        llm_client_->config().model);
    spdlog::info("Tunnel: {}", !config_.relay_url.empty() ? "configured" : "not configured");
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

        // Process tunnel events (syscalls from remote agents)
        process_tunnel_events();

        // Reap dead agents and queue restarts if needed
        agent_manager_->reap_and_restart_agents();

        // Process any pending restarts (with backoff)
        agent_manager_->process_pending_restarts();
    }

    spdlog::info("Kernel shutting down...");
    tunnel_client_->shutdown();
    agent_manager_->stop_all();
    socket_server_->stop();
    spdlog::info("Kernel stopped");
}

void Kernel::shutdown() {
    running_ = false;
}

std::string Kernel::get_llm_model() const {
    if (llm_client_) {
        return llm_client_->config().model;
    }
    return config_.llm_model;
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

        case ipc::SyscallOp::SYS_EXEC:
            return handle_exec(msg);

        case ipc::SyscallOp::SYS_READ:
            return handle_read(msg);

        case ipc::SyscallOp::SYS_WRITE:
            return handle_write(msg);

        // IPC syscalls
        case ipc::SyscallOp::SYS_SEND:
            return handle_send(msg);

        case ipc::SyscallOp::SYS_RECV:
            return handle_recv(msg);

        case ipc::SyscallOp::SYS_BROADCAST:
            return handle_broadcast(msg);

        case ipc::SyscallOp::SYS_REGISTER:
            return handle_register(msg);

        // State Store syscalls
        case ipc::SyscallOp::SYS_STORE:
            return handle_store(msg);

        case ipc::SyscallOp::SYS_FETCH:
            return handle_fetch(msg);

        case ipc::SyscallOp::SYS_DELETE:
            return handle_delete(msg);

        case ipc::SyscallOp::SYS_KEYS:
            return handle_keys(msg);

        // Permission syscalls
        case ipc::SyscallOp::SYS_GET_PERMS:
            return handle_get_perms(msg);

        case ipc::SyscallOp::SYS_SET_PERMS:
            return handle_set_perms(msg);

        // Network syscalls
        case ipc::SyscallOp::SYS_HTTP:
            return handle_http(msg);

        // Event syscalls
        case ipc::SyscallOp::SYS_SUBSCRIBE:
            return handle_subscribe(msg);

        case ipc::SyscallOp::SYS_UNSUBSCRIBE:
            return handle_unsubscribe(msg);

        case ipc::SyscallOp::SYS_POLL_EVENTS:
            return handle_poll_events(msg);

        case ipc::SyscallOp::SYS_EMIT:
            return handle_emit(msg);

        // World simulation syscalls
        case ipc::SyscallOp::SYS_WORLD_CREATE:
            return handle_world_create(msg);

        case ipc::SyscallOp::SYS_WORLD_DESTROY:
            return handle_world_destroy(msg);

        case ipc::SyscallOp::SYS_WORLD_LIST:
            return handle_world_list(msg);

        case ipc::SyscallOp::SYS_WORLD_JOIN:
            return handle_world_join(msg);

        case ipc::SyscallOp::SYS_WORLD_LEAVE:
            return handle_world_leave(msg);

        case ipc::SyscallOp::SYS_WORLD_EVENT:
            return handle_world_event(msg);

        case ipc::SyscallOp::SYS_WORLD_STATE:
            return handle_world_state(msg);

        case ipc::SyscallOp::SYS_WORLD_SNAPSHOT:
            return handle_world_snapshot(msg);

        case ipc::SyscallOp::SYS_WORLD_RESTORE:
            return handle_world_restore(msg);

        // Tunnel syscalls
        case ipc::SyscallOp::SYS_TUNNEL_CONNECT:
            return handle_tunnel_connect(msg);

        case ipc::SyscallOp::SYS_TUNNEL_DISCONNECT:
            return handle_tunnel_disconnect(msg);

        case ipc::SyscallOp::SYS_TUNNEL_STATUS:
            return handle_tunnel_status(msg);

        case ipc::SyscallOp::SYS_TUNNEL_LIST_REMOTES:
            return handle_tunnel_list_remotes(msg);

        case ipc::SyscallOp::SYS_TUNNEL_CONFIG:
            return handle_tunnel_config(msg);

        // Metrics syscalls
        case ipc::SyscallOp::SYS_METRICS_SYSTEM:
            return handle_metrics_system(msg);

        case ipc::SyscallOp::SYS_METRICS_AGENT:
            return handle_metrics_agent(msg);

        case ipc::SyscallOp::SYS_METRICS_ALL_AGENTS:
            return handle_metrics_all_agents(msg);

        case ipc::SyscallOp::SYS_METRICS_CGROUP:
            return handle_metrics_cgroup(msg);

        default:
            spdlog::warn("Unknown opcode: 0x{:02x}", static_cast<uint8_t>(msg.opcode));
            return ipc::Message(msg.agent_id, msg.opcode, msg.payload);
    }
}

ipc::Message Kernel::handle_think(const ipc::Message& msg) {
    auto& perms = get_agent_permissions(msg.agent_id);

    // Check if LLM is configured
    if (!llm_client_->is_configured()) {
        json response;
        response["success"] = false;
        response["error"] = "LLM not configured (set GEMINI_API_KEY)";
        response["content"] = "";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_THINK, response.dump());
    }

    // Check LLM permission and quota
    if (!perms.can_use_llm()) {
        spdlog::warn("Agent {} denied LLM access (quota exceeded or permission denied)", msg.agent_id);
        json response;
        response["success"] = false;
        response["error"] = "Permission denied: LLM quota exceeded or not allowed";
        response["content"] = "";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_THINK, response.dump());
    }

    std::string payload = msg.payload_str();
    spdlog::debug("Agent {} thinking: {}", msg.agent_id,
        payload.length() > 50 ? payload.substr(0, 50) + "..." : payload);

    // Call LLM service with extended options (supports JSON payload)
    auto result = llm_client_->complete_with_options(payload);

    // Track tokens in agent (if successful)
    if (result.success && result.tokens_used > 0) {
        auto agent = agent_manager_->get_agent(msg.agent_id);
        if (agent) {
            agent->record_llm_call(result.tokens_used);
        }
        // Also track in permissions for quota enforcement
        perms.record_llm_usage(result.tokens_used);
    }

    json response;
    response["success"] = result.success;
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

        // Restart configuration
        config.restart.policy = runtime::restart_policy_from_string(
            j.value("restart_policy", "never"));
        config.restart.max_restarts = j.value("max_restarts", 5);
        config.restart.restart_window_sec = j.value("restart_window", 300);

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

        // Track parent-child relationship
        agent->set_parent_id(msg.agent_id);

        // Update parent's child list (if parent exists)
        if (msg.agent_id > 0) {
            auto parent = agent_manager_->get_agent(msg.agent_id);
            if (parent) {
                parent->add_child(agent->id());
            }
        }

        json response;
        response["id"] = agent->id();
        response["name"] = agent->name();
        response["pid"] = agent->pid();
        response["status"] = "running";
        response["restart_policy"] = runtime::restart_policy_to_string(config.restart.policy);

        // Emit AGENT_SPAWNED event
        json event_data;
        event_data["agent_id"] = agent->id();
        event_data["name"] = agent->name();
        event_data["pid"] = agent->pid();
        event_data["parent_id"] = msg.agent_id;
        emit_event(KernelEventType::AGENT_SPAWNED, event_data, 0);

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
        uint32_t target_id = 0;
        std::string target_name;

        // Get agent info before killing
        if (j.contains("id")) {
            target_id = j["id"].get<uint32_t>();
            auto agent = agent_manager_->get_agent(target_id);
            if (agent) target_name = agent->name();
            killed = agent_manager_->kill_agent(target_id);
        } else if (j.contains("name")) {
            target_name = j["name"].get<std::string>();
            auto agent = agent_manager_->get_agent(target_name);
            if (agent) target_id = agent->id();
            killed = agent_manager_->kill_agent(target_name);
        }

        // Emit AGENT_EXITED event if killed
        if (killed && target_id > 0) {
            json event_data;
            event_data["agent_id"] = target_id;
            event_data["name"] = target_name;
            event_data["killed_by"] = msg.agent_id;
            emit_event(KernelEventType::AGENT_EXITED, event_data, 0);
        }

        json response;
        response["killed"] = killed;
        response["agent_id"] = target_id;

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

ipc::Message Kernel::handle_exec(const ipc::Message& msg) {
    auto& perms = get_agent_permissions(msg.agent_id);

    try {
        json j = json::parse(msg.payload_str());
        std::string command = j.value("command", "");
        std::string cwd = j.value("cwd", "");
        int timeout_sec = j.value("timeout", 30);

        if (command.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "command required";
            response["stdout"] = "";
            response["stderr"] = "";
            response["exit_code"] = -1;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EXEC, response.dump());
        }

        // Check permission
        if (!perms.can_execute_command(command)) {
            spdlog::warn("Agent {} denied exec: {}", msg.agent_id, command);
            json response;
            response["success"] = false;
            response["error"] = "Permission denied: command not allowed";
            response["stdout"] = "";
            response["stderr"] = "";
            response["exit_code"] = -1;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EXEC, response.dump());
        }

        spdlog::debug("Agent {} executing: {}", msg.agent_id, command);

        // Build the full command (with optional cwd)
        std::string full_command = command;
        if (!cwd.empty()) {
            full_command = "cd " + cwd + " && " + command;
        }
        // Redirect stderr to stdout with marker
        full_command += " 2>&1";

        // Execute via popen
        FILE* pipe = popen(full_command.c_str(), "r");
        if (!pipe) {
            json response;
            response["success"] = false;
            response["error"] = "failed to execute command";
            response["stdout"] = "";
            response["stderr"] = "";
            response["exit_code"] = -1;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EXEC, response.dump());
        }

        // Read output
        std::string output;
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }

        // Get exit code
        int status = pclose(pipe);
        int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

        json response;
        response["success"] = (exit_code == 0);
        response["stdout"] = output;
        response["stderr"] = "";  // Combined with stdout
        response["exit_code"] = exit_code;

        spdlog::debug("Command exit code: {}", exit_code);
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EXEC, response.dump());

    } catch (const std::exception& e) {
        spdlog::error("Failed to parse exec request: {}", e.what());
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        response["stdout"] = "";
        response["stderr"] = "";
        response["exit_code"] = -1;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EXEC, response.dump());
    }
}

ipc::Message Kernel::handle_read(const ipc::Message& msg) {
    // Check if agent is in a world - route through VFS if so
    if (world_engine_->is_agent_in_world(msg.agent_id)) {
        auto world_id = world_engine_->get_agent_world(msg.agent_id);
        if (world_id) {
            auto* world = world_engine_->get_world(*world_id);
            if (world && world->vfs().is_enabled()) {
                // Parse to check if VFS should intercept
                try {
                    json j = json::parse(msg.payload_str());
                    std::string path = j.value("path", "");
                    if (world->vfs().should_intercept(path)) {
                        return handle_read_virtual(msg, world);
                    }
                } catch (...) {
                    // Fall through to normal handling
                }
            }
        }
    }

    auto& perms = get_agent_permissions(msg.agent_id);

    try {
        json j = json::parse(msg.payload_str());
        std::string path = j.value("path", "");

        if (path.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "path required";
            response["content"] = "";
            response["size"] = 0;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_READ, response.dump());
        }

        // Check permission
        if (!perms.can_read_path(path)) {
            spdlog::warn("Agent {} denied read access to: {}", msg.agent_id, path);
            json response;
            response["success"] = false;
            response["error"] = "Permission denied: path not allowed for reading";
            response["content"] = "";
            response["size"] = 0;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_READ, response.dump());
        }

        spdlog::debug("Agent {} reading file: {}", msg.agent_id, path);

        // Open and read file
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            json response;
            response["success"] = false;
            response["error"] = "failed to open file";
            response["content"] = "";
            response["size"] = 0;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_READ, response.dump());
        }

        // Get file size
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Read content
        std::string content(size, '\0');
        file.read(&content[0], size);
        file.close();

        json response;
        response["success"] = true;
        response["content"] = content;
        response["size"] = size;

        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_READ, response.dump());

    } catch (const std::exception& e) {
        spdlog::error("Failed to parse read request: {}", e.what());
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        response["content"] = "";
        response["size"] = 0;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_READ, response.dump());
    }
}

ipc::Message Kernel::handle_write(const ipc::Message& msg) {
    // Check if agent is in a world - route through VFS if so
    if (world_engine_->is_agent_in_world(msg.agent_id)) {
        auto world_id = world_engine_->get_agent_world(msg.agent_id);
        if (world_id) {
            auto* world = world_engine_->get_world(*world_id);
            if (world && world->vfs().is_enabled()) {
                // Parse to check if VFS should intercept
                try {
                    json j = json::parse(msg.payload_str());
                    std::string path = j.value("path", "");
                    if (world->vfs().should_intercept(path)) {
                        return handle_write_virtual(msg, world);
                    }
                } catch (...) {
                    // Fall through to normal handling
                }
            }
        }
    }

    auto& perms = get_agent_permissions(msg.agent_id);

    try {
        json j = json::parse(msg.payload_str());
        std::string path = j.value("path", "");
        std::string content = j.value("content", "");
        std::string mode = j.value("mode", "write");  // "write" or "append"

        if (path.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "path required";
            response["bytes_written"] = 0;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WRITE, response.dump());
        }

        // Check permission
        if (!perms.can_write_path(path)) {
            spdlog::warn("Agent {} denied write access to: {}", msg.agent_id, path);
            json response;
            response["success"] = false;
            response["error"] = "Permission denied: path not allowed for writing";
            response["bytes_written"] = 0;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WRITE, response.dump());
        }

        spdlog::debug("Agent {} writing file: {} (mode={})", msg.agent_id, path, mode);

        // Open file with appropriate mode
        std::ios_base::openmode file_mode = std::ios::binary;
        if (mode == "append") {
            file_mode |= std::ios::app;
        } else {
            file_mode |= std::ios::trunc;
        }

        std::ofstream file(path, file_mode);
        if (!file.is_open()) {
            json response;
            response["success"] = false;
            response["error"] = "failed to open file for writing";
            response["bytes_written"] = 0;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WRITE, response.dump());
        }

        // Write content
        file.write(content.data(), content.size());
        file.close();

        json response;
        response["success"] = true;
        response["bytes_written"] = content.size();

        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WRITE, response.dump());

    } catch (const std::exception& e) {
        spdlog::error("Failed to parse write request: {}", e.what());
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        response["bytes_written"] = 0;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WRITE, response.dump());
    }
}

// ============================================================================
// IPC Handlers - Inter-Agent Communication
// ============================================================================

ipc::Message Kernel::handle_register(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());
        std::string name = j.value("name", "");

        if (name.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "name required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_REGISTER, response.dump());
        }

        {
            std::lock_guard<std::mutex> lock(registry_mutex_);

            // Check if name already taken
            if (agent_names_.count(name) > 0 && agent_names_[name] != msg.agent_id) {
                json response;
                response["success"] = false;
                response["error"] = "name already registered";
                return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_REGISTER, response.dump());
            }

            // Register the name
            agent_names_[name] = msg.agent_id;
            agent_ids_to_names_[msg.agent_id] = name;
        }

        spdlog::info("Agent {} registered as '{}'", msg.agent_id, name);

        json response;
        response["success"] = true;
        response["agent_id"] = msg.agent_id;
        response["name"] = name;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_REGISTER, response.dump());

    } catch (const std::exception& e) {
        spdlog::error("Failed to parse register request: {}", e.what());
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_REGISTER, response.dump());
    }
}

ipc::Message Kernel::handle_send(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        // Determine target agent
        uint32_t target_id = j.value("to", 0u);
        std::string target_name = j.value("to_name", "");
        json message_content = j.value("message", json::object());

        // Resolve name to ID if needed
        if (target_id == 0 && !target_name.empty()) {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            if (agent_names_.count(target_name) > 0) {
                target_id = agent_names_[target_name];
            } else {
                json response;
                response["success"] = false;
                response["error"] = "target agent not found: " + target_name;
                return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SEND, response.dump());
            }
        }

        if (target_id == 0) {
            json response;
            response["success"] = false;
            response["error"] = "target agent required (to or to_name)";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SEND, response.dump());
        }

        // Get sender's name
        std::string sender_name;
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            if (agent_ids_to_names_.count(msg.agent_id) > 0) {
                sender_name = agent_ids_to_names_[msg.agent_id];
            }
        }

        // Create IPC message
        IPCMessage ipc_msg;
        ipc_msg.from_id = msg.agent_id;
        ipc_msg.from_name = sender_name;
        ipc_msg.message = message_content;
        ipc_msg.timestamp = std::chrono::steady_clock::now();

        // Queue message for target
        {
            std::lock_guard<std::mutex> lock(mailbox_mutex_);
            agent_mailboxes_[target_id].push(ipc_msg);
        }

        spdlog::debug("Agent {} sent message to agent {}", msg.agent_id, target_id);

        json response;
        response["success"] = true;
        response["delivered_to"] = target_id;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SEND, response.dump());

    } catch (const std::exception& e) {
        spdlog::error("Failed to parse send request: {}", e.what());
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SEND, response.dump());
    }
}

ipc::Message Kernel::handle_recv(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());
        int max_messages = j.value("max", 10);  // Default: retrieve up to 10 messages
        bool wait = j.value("wait", false);     // Non-blocking by default

        json messages_array = json::array();

        {
            std::lock_guard<std::mutex> lock(mailbox_mutex_);

            auto& mailbox = agent_mailboxes_[msg.agent_id];
            int count = 0;

            while (!mailbox.empty() && count < max_messages) {
                const IPCMessage& ipc_msg = mailbox.front();

                // Calculate time since message was sent
                auto now = std::chrono::steady_clock::now();
                auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - ipc_msg.timestamp).count();

                json msg_json;
                msg_json["from"] = ipc_msg.from_id;
                msg_json["from_name"] = ipc_msg.from_name;
                msg_json["message"] = ipc_msg.message;
                msg_json["age_ms"] = age_ms;

                messages_array.push_back(msg_json);
                mailbox.pop();
                count++;
            }
        }

        json response;
        response["success"] = true;
        response["messages"] = messages_array;
        response["count"] = messages_array.size();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_RECV, response.dump());

    } catch (const std::exception& e) {
        spdlog::error("Failed to parse recv request: {}", e.what());
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        response["messages"] = json::array();
        response["count"] = 0;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_RECV, response.dump());
    }
}

ipc::Message Kernel::handle_broadcast(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());
        json message_content = j.value("message", json::object());
        bool include_self = j.value("include_self", false);

        // Get sender's name
        std::string sender_name;
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            if (agent_ids_to_names_.count(msg.agent_id) > 0) {
                sender_name = agent_ids_to_names_[msg.agent_id];
            }
        }

        // Create IPC message
        IPCMessage ipc_msg;
        ipc_msg.from_id = msg.agent_id;
        ipc_msg.from_name = sender_name;
        ipc_msg.message = message_content;
        ipc_msg.timestamp = std::chrono::steady_clock::now();

        // Get all registered agents and queue message
        int delivered_count = 0;
        {
            std::lock_guard<std::mutex> reg_lock(registry_mutex_);
            std::lock_guard<std::mutex> mail_lock(mailbox_mutex_);

            for (const auto& [agent_id, name] : agent_ids_to_names_) {
                // Skip sender unless include_self is true
                if (agent_id == msg.agent_id && !include_self) {
                    continue;
                }

                agent_mailboxes_[agent_id].push(ipc_msg);
                delivered_count++;
            }
        }

        spdlog::debug("Agent {} broadcast message to {} agents", msg.agent_id, delivered_count);

        json response;
        response["success"] = true;
        response["delivered_count"] = delivered_count;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_BROADCAST, response.dump());

    } catch (const std::exception& e) {
        spdlog::error("Failed to parse broadcast request: {}", e.what());
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        response["delivered_count"] = 0;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_BROADCAST, response.dump());
    }
}

// ============================================================================
// Permission Handlers
// ============================================================================

AgentPermissions& Kernel::get_agent_permissions(uint32_t agent_id) {
    std::lock_guard<std::mutex> lock(permissions_mutex_);

    auto it = agent_permissions_.find(agent_id);
    if (it == agent_permissions_.end()) {
        // Create default permissions for new agent (STANDARD level)
        agent_permissions_[agent_id] = AgentPermissions::from_level(PermissionLevel::STANDARD);
    }
    return agent_permissions_[agent_id];
}

ipc::Message Kernel::handle_get_perms(const ipc::Message& msg) {
    auto& perms = get_agent_permissions(msg.agent_id);

    json response;
    response["success"] = true;
    response["permissions"] = perms.to_json();

    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_GET_PERMS, response.dump());
}

ipc::Message Kernel::handle_set_perms(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        uint32_t target_id = j.value("agent_id", msg.agent_id);
        auto& caller_perms = get_agent_permissions(msg.agent_id);

        // Only agents with can_spawn permission can set other agents' permissions
        if (target_id != msg.agent_id && !caller_perms.can_spawn) {
            json response;
            response["success"] = false;
            response["error"] = "Permission denied: cannot modify other agent's permissions";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SET_PERMS, response.dump());
        }

        // Parse and set new permissions
        if (j.contains("permissions")) {
            std::lock_guard<std::mutex> lock(permissions_mutex_);
            agent_permissions_[target_id] = AgentPermissions::from_json(j["permissions"]);
            spdlog::info("Agent {} set permissions for agent {}", msg.agent_id, target_id);
        } else if (j.contains("level")) {
            std::string level_str = j["level"].get<std::string>();
            PermissionLevel level = PermissionLevel::STANDARD;

            if (level_str == "unrestricted") level = PermissionLevel::UNRESTRICTED;
            else if (level_str == "standard") level = PermissionLevel::STANDARD;
            else if (level_str == "sandboxed") level = PermissionLevel::SANDBOXED;
            else if (level_str == "readonly") level = PermissionLevel::READONLY;
            else if (level_str == "minimal") level = PermissionLevel::MINIMAL;

            std::lock_guard<std::mutex> lock(permissions_mutex_);
            agent_permissions_[target_id] = AgentPermissions::from_level(level);
            spdlog::info("Agent {} set permission level {} for agent {}", msg.agent_id, level_str, target_id);
        }

        json response;
        response["success"] = true;
        response["agent_id"] = target_id;

        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SET_PERMS, response.dump());

    } catch (const std::exception& e) {
        spdlog::error("Failed to parse set_perms request: {}", e.what());
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SET_PERMS, response.dump());
    }
}

// ============================================================================
// HTTP Handler
// ============================================================================

ipc::Message Kernel::handle_http(const ipc::Message& msg) {
    // Check if agent is in a world - route through network mock if so
    if (world_engine_->is_agent_in_world(msg.agent_id)) {
        auto world_id = world_engine_->get_agent_world(msg.agent_id);
        if (world_id) {
            auto* world = world_engine_->get_world(*world_id);
            if (world && world->network().is_enabled()) {
                // Parse to check if network mock should intercept
                try {
                    json j = json::parse(msg.payload_str());
                    std::string url = j.value("url", "");
                    if (world->network().should_intercept(url)) {
                        return handle_http_virtual(msg, world);
                    }
                } catch (...) {
                    // Fall through to normal handling
                }
            }
        }
    }

    auto& perms = get_agent_permissions(msg.agent_id);

    try {
        json j = json::parse(msg.payload_str());
        std::string method = j.value("method", "GET");
        std::string url = j.value("url", "");
        int timeout = j.value("timeout", 30);

        if (url.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "URL required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }

        // Check HTTP permission
        if (!perms.can_http) {
            spdlog::warn("Agent {} denied HTTP access (permission denied)", msg.agent_id);
            json response;
            response["success"] = false;
            response["error"] = "Permission denied: HTTP not allowed";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }

        // Check domain whitelist
        std::string domain = PermissionChecker::extract_domain(url);
        if (!perms.can_access_domain(domain)) {
            spdlog::warn("Agent {} denied access to domain: {}", msg.agent_id, domain);
            json response;
            response["success"] = false;
            response["error"] = "Permission denied: domain not in whitelist: " + domain;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }

        spdlog::debug("Agent {} making {} request to {}", msg.agent_id, method, url);

        // Build curl command
        std::string curl_cmd = "curl -s -X " + method;
        curl_cmd += " --max-time " + std::to_string(timeout);

        // Add headers if present
        if (j.contains("headers") && j["headers"].is_object()) {
            for (auto& [key, val] : j["headers"].items()) {
                curl_cmd += " -H '" + key + ": " + val.get<std::string>() + "'";
            }
        }

        // Add body if present (for POST/PUT)
        if (j.contains("body") && (method == "POST" || method == "PUT" || method == "PATCH")) {
            std::string body = j["body"].get<std::string>();
            curl_cmd += " -d '" + body + "'";
        }

        curl_cmd += " '" + url + "' 2>&1";

        // Execute curl
        FILE* pipe = popen(curl_cmd.c_str(), "r");
        if (!pipe) {
            json response;
            response["success"] = false;
            response["error"] = "Failed to execute HTTP request";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }

        std::string output;
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }

        int status = pclose(pipe);
        bool success = (WIFEXITED(status) && WEXITSTATUS(status) == 0);

        json response;
        response["success"] = success;
        response["body"] = output;
        response["status_code"] = success ? 200 : 0;  // Simplified - curl doesn't give us status code easily

        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());

    } catch (const std::exception& e) {
        spdlog::error("Failed to parse HTTP request: {}", e.what());
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
    }
}

// State Store: check if agent can access a key based on scope
bool Kernel::can_access_key(uint32_t agent_id, const std::string& key, const StoredValue& value) const {
    if (value.scope == "global") return true;
    if (value.scope == "agent" && value.owner_agent_id == agent_id) return true;
    if (value.scope == "session") return true;  // Session scope = accessible to all in this kernel session
    return false;
}

// State Store: STORE syscall
ipc::Message Kernel::handle_store(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::string key = j.value("key", "");
        if (key.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "key is required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_STORE, response.dump());
        }

        StoredValue entry;
        entry.value = j.value("value", json{});
        entry.owner_agent_id = msg.agent_id;
        entry.scope = j.value("scope", "global");

        // Handle TTL
        if (j.contains("ttl") && j["ttl"].is_number()) {
            int ttl_secs = j["ttl"].get<int>();
            entry.expires_at = std::chrono::steady_clock::now() + std::chrono::seconds(ttl_secs);
        }

        // Validate scope
        if (entry.scope != "global" && entry.scope != "agent" && entry.scope != "session") {
            entry.scope = "global";
        }

        // For agent scope, prefix the key with agent_id
        std::string store_key = key;
        if (entry.scope == "agent") {
            store_key = "agent:" + std::to_string(msg.agent_id) + ":" + key;
        }

        {
            std::lock_guard<std::mutex> lock(state_store_mutex_);
            state_store_[store_key] = std::move(entry);
        }

        spdlog::debug("Agent {} stored key '{}' (scope={})", msg.agent_id, key, entry.scope);

        // Emit STATE_CHANGED event for global scope keys
        if (entry.scope == "global") {
            json event_data;
            event_data["key"] = key;
            event_data["action"] = "store";
            event_data["agent_id"] = msg.agent_id;
            emit_event(KernelEventType::STATE_CHANGED, event_data, msg.agent_id);
        }

        json response;
        response["success"] = true;
        response["key"] = key;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_STORE, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_STORE, response.dump());
    }
}

// State Store: FETCH syscall
ipc::Message Kernel::handle_fetch(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::string key = j.value("key", "");
        if (key.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "key is required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_FETCH, response.dump());
        }

        // Try both global key and agent-scoped key
        std::vector<std::string> keys_to_try = {
            key,
            "agent:" + std::to_string(msg.agent_id) + ":" + key
        };

        std::lock_guard<std::mutex> lock(state_store_mutex_);

        for (const auto& try_key : keys_to_try) {
            auto it = state_store_.find(try_key);
            if (it != state_store_.end()) {
                // Check expiration
                if (it->second.is_expired()) {
                    state_store_.erase(it);
                    continue;
                }

                // Check access
                if (!can_access_key(msg.agent_id, try_key, it->second)) {
                    continue;
                }

                json response;
                response["success"] = true;
                response["exists"] = true;
                response["value"] = it->second.value;
                response["scope"] = it->second.scope;
                return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_FETCH, response.dump());
            }
        }

        // Key not found
        json response;
        response["success"] = true;
        response["exists"] = false;
        response["value"] = nullptr;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_FETCH, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_FETCH, response.dump());
    }
}

// State Store: DELETE syscall
ipc::Message Kernel::handle_delete(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::string key = j.value("key", "");
        if (key.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "key is required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_DELETE, response.dump());
        }

        // Try both global key and agent-scoped key
        std::vector<std::string> keys_to_try = {
            key,
            "agent:" + std::to_string(msg.agent_id) + ":" + key
        };

        std::lock_guard<std::mutex> lock(state_store_mutex_);

        bool deleted = false;
        for (const auto& try_key : keys_to_try) {
            auto it = state_store_.find(try_key);
            if (it != state_store_.end()) {
                // Only owner can delete (or global scope)
                if (it->second.owner_agent_id == msg.agent_id || it->second.scope == "global") {
                    state_store_.erase(it);
                    deleted = true;
                    spdlog::debug("Agent {} deleted key '{}'", msg.agent_id, key);
                    break;
                }
            }
        }

        json response;
        response["success"] = true;
        response["deleted"] = deleted;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_DELETE, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_DELETE, response.dump());
    }
}

// State Store: KEYS syscall
ipc::Message Kernel::handle_keys(const ipc::Message& msg) {
    try {
        json j;
        if (!msg.payload.empty()) {
            j = json::parse(msg.payload_str());
        }

        std::string prefix = j.value("prefix", "");

        std::lock_guard<std::mutex> lock(state_store_mutex_);

        std::vector<std::string> keys;
        for (auto it = state_store_.begin(); it != state_store_.end(); ) {
            // Clean up expired entries
            if (it->second.is_expired()) {
                it = state_store_.erase(it);
                continue;
            }

            // Check access
            if (!can_access_key(msg.agent_id, it->first, it->second)) {
                ++it;
                continue;
            }

            // Check prefix match
            const std::string& key = it->first;
            if (prefix.empty() || key.find(prefix) == 0) {
                // For agent-scoped keys, strip the prefix when returning
                if (key.find("agent:") == 0) {
                    size_t second_colon = key.find(':', 6);
                    if (second_colon != std::string::npos) {
                        keys.push_back(key.substr(second_colon + 1));
                    } else {
                        keys.push_back(key);
                    }
                } else {
                    keys.push_back(key);
                }
            }
            ++it;
        }

        json response;
        response["success"] = true;
        response["keys"] = keys;
        response["count"] = keys.size();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_KEYS, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_KEYS, response.dump());
    }
}

// Events: emit an event to all subscribed agents
void Kernel::emit_event(KernelEventType type, const nlohmann::json& data, uint32_t source_agent_id) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    KernelEvent event;
    event.type = type;
    event.data = data;
    event.timestamp = std::chrono::steady_clock::now();
    event.source_agent_id = source_agent_id;

    // Deliver to all subscribed agents
    for (const auto& [agent_id, subscriptions] : event_subscriptions_) {
        if (subscriptions.count(type) > 0) {
            event_queues_[agent_id].push(event);
            spdlog::debug("Event {} queued for agent {}", kernel_event_type_to_string(type), agent_id);
        }
    }
}

// Events: SUBSCRIBE syscall
ipc::Message Kernel::handle_subscribe(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::vector<std::string> event_types;
        // Accept both "events" and "event_types" keys for compatibility
        if (j.contains("event_types") && j["event_types"].is_array()) {
            for (const auto& e : j["event_types"]) {
                event_types.push_back(e.get<std::string>());
            }
        } else if (j.contains("events") && j["events"].is_array()) {
            for (const auto& e : j["events"]) {
                event_types.push_back(e.get<std::string>());
            }
        } else if (j.contains("event")) {
            event_types.push_back(j["event"].get<std::string>());
        }

        if (event_types.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "No events specified";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SUBSCRIBE, response.dump());
        }

        std::lock_guard<std::mutex> lock(events_mutex_);

        auto& subs = event_subscriptions_[msg.agent_id];
        for (const auto& event_str : event_types) {
            KernelEventType type = kernel_event_type_from_string(event_str);
            subs.insert(type);
        }

        spdlog::debug("Agent {} subscribed to {} event type(s)", msg.agent_id, event_types.size());

        json response;
        response["success"] = true;
        response["subscribed"] = event_types;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SUBSCRIBE, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SUBSCRIBE, response.dump());
    }
}

// Events: UNSUBSCRIBE syscall
ipc::Message Kernel::handle_unsubscribe(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::vector<std::string> event_types;
        bool unsubscribe_all = j.value("all", false);

        if (!unsubscribe_all) {
            // Accept both "events" and "event_types" keys for compatibility
            if (j.contains("event_types") && j["event_types"].is_array()) {
                for (const auto& e : j["event_types"]) {
                    event_types.push_back(e.get<std::string>());
                }
            } else if (j.contains("events") && j["events"].is_array()) {
                for (const auto& e : j["events"]) {
                    event_types.push_back(e.get<std::string>());
                }
            } else if (j.contains("event")) {
                event_types.push_back(j["event"].get<std::string>());
            }
        }

        std::lock_guard<std::mutex> lock(events_mutex_);

        if (unsubscribe_all) {
            event_subscriptions_.erase(msg.agent_id);
            spdlog::debug("Agent {} unsubscribed from all events", msg.agent_id);
        } else {
            auto it = event_subscriptions_.find(msg.agent_id);
            if (it != event_subscriptions_.end()) {
                for (const auto& event_str : event_types) {
                    KernelEventType type = kernel_event_type_from_string(event_str);
                    it->second.erase(type);
                }
            }
            spdlog::debug("Agent {} unsubscribed from {} event type(s)", msg.agent_id, event_types.size());
        }

        json response;
        response["success"] = true;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_UNSUBSCRIBE, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_UNSUBSCRIBE, response.dump());
    }
}

// Events: POLL_EVENTS syscall
ipc::Message Kernel::handle_poll_events(const ipc::Message& msg) {
    try {
        json j;
        if (!msg.payload.empty()) {
            j = json::parse(msg.payload_str());
        }

        int max_events = j.value("max", 100);

        std::lock_guard<std::mutex> lock(events_mutex_);

        json events_array = json::array();
        auto it = event_queues_.find(msg.agent_id);

        if (it != event_queues_.end()) {
            auto& queue = it->second;
            int count = 0;

            while (!queue.empty() && count < max_events) {
                const auto& event = queue.front();

                json event_json;
                event_json["type"] = kernel_event_type_to_string(event.type);
                event_json["data"] = event.data;
                event_json["source_agent_id"] = event.source_agent_id;

                // Convert timestamp to milliseconds since epoch
                auto duration = event.timestamp.time_since_epoch();
                auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
                event_json["timestamp"] = millis;

                events_array.push_back(event_json);
                queue.pop();
                count++;
            }
        }

        json response;
        response["success"] = true;
        response["events"] = events_array;
        response["count"] = events_array.size();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_POLL_EVENTS, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_POLL_EVENTS, response.dump());
    }
}

// Events: EMIT syscall (custom events from agents)
ipc::Message Kernel::handle_emit(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::string event_name = j.value("event", "CUSTOM");
        json event_data = j.value("data", json{});

        // Only allow CUSTOM events from agents
        KernelEventType type = KernelEventType::CUSTOM;
        if (event_name != "CUSTOM") {
            event_data["custom_type"] = event_name;
        }

        emit_event(type, event_data, msg.agent_id);

        spdlog::debug("Agent {} emitted event: {}", msg.agent_id, event_name);

        json response;
        response["success"] = true;
        response["event"] = event_name;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EMIT, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EMIT, response.dump());
    }
}

// ============================================================================
// World Simulation Handlers
// ============================================================================

ipc::Message Kernel::handle_world_create(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::string name = j.value("name", "unnamed");
        json config = j.value("config", json::object());

        auto world_id = world_engine_->create_world(name, config);
        if (!world_id) {
            json response;
            response["success"] = false;
            response["error"] = "Failed to create world";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_CREATE, response.dump());
        }

        spdlog::info("Agent {} created world '{}' (name={})", msg.agent_id, *world_id, name);

        json response;
        response["success"] = true;
        response["world_id"] = *world_id;
        response["name"] = name;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_CREATE, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_CREATE, response.dump());
    }
}

ipc::Message Kernel::handle_world_destroy(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::string world_id = j.value("world_id", "");
        bool force = j.value("force", false);

        if (world_id.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "world_id required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_DESTROY, response.dump());
        }

        bool destroyed = world_engine_->destroy_world(world_id, force);

        if (!destroyed) {
            json response;
            response["success"] = false;
            response["error"] = "Failed to destroy world (not found or has active agents)";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_DESTROY, response.dump());
        }

        spdlog::info("Agent {} destroyed world '{}'", msg.agent_id, world_id);

        json response;
        response["success"] = true;
        response["world_id"] = world_id;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_DESTROY, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_DESTROY, response.dump());
    }
}

ipc::Message Kernel::handle_world_list(const ipc::Message& msg) {
    auto worlds = world_engine_->list_worlds();

    json response;
    response["success"] = true;
    response["worlds"] = worlds;
    response["count"] = worlds.size();
    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_LIST, response.dump());
}

ipc::Message Kernel::handle_world_join(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::string world_id = j.value("world_id", "");

        if (world_id.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "world_id required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_JOIN, response.dump());
        }

        bool joined = world_engine_->join_world(msg.agent_id, world_id);

        if (!joined) {
            json response;
            response["success"] = false;
            response["error"] = "Failed to join world (not found or already in a world)";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_JOIN, response.dump());
        }

        spdlog::info("Agent {} joined world '{}'", msg.agent_id, world_id);

        json response;
        response["success"] = true;
        response["world_id"] = world_id;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_JOIN, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_JOIN, response.dump());
    }
}

ipc::Message Kernel::handle_world_leave(const ipc::Message& msg) {
    bool left = world_engine_->leave_world(msg.agent_id);

    if (!left) {
        json response;
        response["success"] = false;
        response["error"] = "Not in any world";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_LEAVE, response.dump());
    }

    spdlog::info("Agent {} left world", msg.agent_id);

    json response;
    response["success"] = true;
    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_LEAVE, response.dump());
}

ipc::Message Kernel::handle_world_event(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::string world_id = j.value("world_id", "");
        std::string event_type = j.value("event_type", "");
        json params = j.value("params", json::object());

        if (world_id.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "world_id required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_EVENT, response.dump());
        }

        if (event_type.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "event_type required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_EVENT, response.dump());
        }

        bool injected = world_engine_->inject_event(world_id, event_type, params);

        if (!injected) {
            json response;
            response["success"] = false;
            response["error"] = "Failed to inject event (world not found)";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_EVENT, response.dump());
        }

        spdlog::info("Agent {} injected chaos event '{}' into world '{}'",
                     msg.agent_id, event_type, world_id);

        json response;
        response["success"] = true;
        response["world_id"] = world_id;
        response["event_type"] = event_type;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_EVENT, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_EVENT, response.dump());
    }
}

ipc::Message Kernel::handle_world_state(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::string world_id = j.value("world_id", "");

        if (world_id.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "world_id required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_STATE, response.dump());
        }

        auto state = world_engine_->get_world_state(world_id);

        if (!state) {
            json response;
            response["success"] = false;
            response["error"] = "World not found";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_STATE, response.dump());
        }

        json response;
        response["success"] = true;
        response["state"] = *state;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_STATE, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_STATE, response.dump());
    }
}

ipc::Message Kernel::handle_world_snapshot(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        std::string world_id = j.value("world_id", "");

        if (world_id.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "world_id required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_SNAPSHOT, response.dump());
        }

        auto snapshot = world_engine_->snapshot_world(world_id);

        if (!snapshot) {
            json response;
            response["success"] = false;
            response["error"] = "World not found";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_SNAPSHOT, response.dump());
        }

        spdlog::info("Agent {} created snapshot of world '{}'", msg.agent_id, world_id);

        json response;
        response["success"] = true;
        response["snapshot"] = *snapshot;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_SNAPSHOT, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_SNAPSHOT, response.dump());
    }
}

ipc::Message Kernel::handle_world_restore(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        json snapshot = j.value("snapshot", json{});
        std::string new_world_id = j.value("new_world_id", "");

        if (snapshot.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "snapshot required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_RESTORE, response.dump());
        }

        auto world_id = world_engine_->restore_world(snapshot, new_world_id);

        if (!world_id) {
            json response;
            response["success"] = false;
            response["error"] = "Failed to restore world";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_RESTORE, response.dump());
        }

        spdlog::info("Agent {} restored world as '{}'", msg.agent_id, *world_id);

        json response;
        response["success"] = true;
        response["world_id"] = *world_id;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_RESTORE, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WORLD_RESTORE, response.dump());
    }
}

// ============================================================================
// World-Aware I/O Helpers
// ============================================================================

ipc::Message Kernel::handle_read_virtual(const ipc::Message& msg, World* world) {
    try {
        json j = json::parse(msg.payload_str());
        std::string path = j.value("path", "");

        // Record syscall
        world->record_syscall();

        // Check chaos injection
        if (world->chaos().should_fail_read(path)) {
            spdlog::debug("Chaos: Injected read failure for {} in world '{}'", path, world->id());
            json response;
            response["success"] = false;
            response["error"] = "Simulated I/O failure (chaos)";
            response["content"] = "";
            response["size"] = 0;
            response["world"] = world->id();
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_READ, response.dump());
        }

        // Inject latency if configured
        uint32_t latency = world->chaos().get_latency();
        if (latency > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(latency));
        }

        // Read from VFS
        auto content = world->vfs().read(path);

        if (!content) {
            json response;
            response["success"] = false;
            response["error"] = "File not found in virtual filesystem";
            response["content"] = "";
            response["size"] = 0;
            response["world"] = world->id();
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_READ, response.dump());
        }

        spdlog::debug("Agent {} read {} bytes from VFS path {} in world '{}'",
                      msg.agent_id, content->size(), path, world->id());

        json response;
        response["success"] = true;
        response["content"] = *content;
        response["size"] = content->size();
        response["world"] = world->id();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_READ, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        response["content"] = "";
        response["size"] = 0;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_READ, response.dump());
    }
}

ipc::Message Kernel::handle_write_virtual(const ipc::Message& msg, World* world) {
    try {
        json j = json::parse(msg.payload_str());
        std::string path = j.value("path", "");
        std::string content = j.value("content", "");
        std::string mode = j.value("mode", "write");

        // Record syscall
        world->record_syscall();

        // Check chaos injection
        if (world->chaos().should_fail_write(path)) {
            spdlog::debug("Chaos: Injected write failure for {} in world '{}'", path, world->id());
            json response;
            response["success"] = false;
            response["error"] = "Simulated I/O failure (chaos)";
            response["bytes_written"] = 0;
            response["world"] = world->id();
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WRITE, response.dump());
        }

        // Inject latency if configured
        uint32_t latency = world->chaos().get_latency();
        if (latency > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(latency));
        }

        // Check if path is writable
        if (!world->vfs().is_writable(path)) {
            json response;
            response["success"] = false;
            response["error"] = "Path not writable in virtual filesystem";
            response["bytes_written"] = 0;
            response["world"] = world->id();
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WRITE, response.dump());
        }

        // Write to VFS
        bool append = (mode == "append");
        bool written = world->vfs().write(path, content, append);

        if (!written) {
            json response;
            response["success"] = false;
            response["error"] = "Failed to write to virtual filesystem";
            response["bytes_written"] = 0;
            response["world"] = world->id();
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WRITE, response.dump());
        }

        spdlog::debug("Agent {} wrote {} bytes to VFS path {} in world '{}' (mode={})",
                      msg.agent_id, content.size(), path, world->id(), mode);

        json response;
        response["success"] = true;
        response["bytes_written"] = content.size();
        response["world"] = world->id();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WRITE, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        response["bytes_written"] = 0;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_WRITE, response.dump());
    }
}

ipc::Message Kernel::handle_http_virtual(const ipc::Message& msg, World* world) {
    try {
        json j = json::parse(msg.payload_str());
        std::string method = j.value("method", "GET");
        std::string url = j.value("url", "");

        // Record syscall
        world->record_syscall();

        // Check chaos injection for network
        if (world->chaos().should_fail_network(url)) {
            spdlog::debug("Chaos: Injected network failure for {} in world '{}'", url, world->id());
            json response;
            response["success"] = false;
            response["error"] = "Simulated network failure (chaos)";
            response["body"] = "";
            response["status_code"] = 503;
            response["world"] = world->id();
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }

        // Get mock response
        auto mock_response = world->network().get_response(url, method);

        if (!mock_response) {
            // Passthrough to real network (would need to call original handle_http logic)
            // For now, just return an error indicating no mock available
            json response;
            response["success"] = false;
            response["error"] = "No mock response configured for URL";
            response["body"] = "";
            response["status_code"] = 0;
            response["world"] = world->id();
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }

        // Inject latency from mock response + chaos
        uint32_t total_latency = mock_response->latency_ms + world->chaos().get_latency();
        if (total_latency > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(total_latency));
        }

        spdlog::debug("Agent {} got mock HTTP response for {} in world '{}': status={}",
                      msg.agent_id, url, world->id(), mock_response->status_code);

        json response;
        response["success"] = (mock_response->status_code >= 200 && mock_response->status_code < 400);
        response["body"] = mock_response->body;
        response["status_code"] = mock_response->status_code;
        response["headers"] = mock_response->headers;
        response["world"] = world->id();
        response["mocked"] = true;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        response["body"] = "";
        response["status_code"] = 0;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
    }
}

// ============================================================================
// Tunnel Handlers
// ============================================================================

ipc::Message Kernel::handle_tunnel_connect(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        // Allow overriding config via syscall
        std::string relay_url = j.value("relay_url", config_.relay_url);
        std::string machine_id = j.value("machine_id", config_.machine_id);
        std::string token = j.value("token", config_.machine_token);

        if (relay_url.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "relay_url required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_TUNNEL_CONNECT, response.dump());
        }

        // Configure tunnel
        TunnelConfig tc;
        tc.relay_url = relay_url;
        tc.machine_id = machine_id;
        tc.token = token;
        tunnel_client_->configure(tc);

        // Connect
        if (tunnel_client_->connect()) {
            spdlog::info("Tunnel connected via syscall: {}", relay_url);
            json response;
            response["success"] = true;
            response["relay_url"] = relay_url;
            response["machine_id"] = machine_id;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_TUNNEL_CONNECT, response.dump());
        } else {
            json response;
            response["success"] = false;
            response["error"] = "Failed to connect to relay server";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_TUNNEL_CONNECT, response.dump());
        }

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_TUNNEL_CONNECT, response.dump());
    }
}

ipc::Message Kernel::handle_tunnel_disconnect(const ipc::Message& msg) {
    tunnel_client_->disconnect();

    json response;
    response["success"] = true;
    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_TUNNEL_DISCONNECT, response.dump());
}

ipc::Message Kernel::handle_tunnel_status(const ipc::Message& msg) {
    auto status = tunnel_client_->get_status();

    json response;
    response["success"] = true;
    response["connected"] = status.connected;
    response["relay_url"] = status.relay_url;
    response["machine_id"] = status.machine_id;
    response["remote_agent_count"] = status.remote_agent_count;

    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_TUNNEL_STATUS, response.dump());
}

ipc::Message Kernel::handle_tunnel_list_remotes(const ipc::Message& msg) {
    auto agents = tunnel_client_->list_remote_agents();

    json response;
    response["success"] = true;
    response["agents"] = json::array();

    for (const auto& agent : agents) {
        json a;
        a["agent_id"] = agent.agent_id;
        a["name"] = agent.name;
        a["connected_at"] = agent.connected_at;
        response["agents"].push_back(a);
    }
    response["count"] = agents.size();

    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_TUNNEL_LIST_REMOTES, response.dump());
}

ipc::Message Kernel::handle_tunnel_config(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        TunnelConfig tc;
        tc.relay_url = j.value("relay_url", config_.relay_url);
        tc.machine_id = j.value("machine_id", config_.machine_id);
        tc.token = j.value("token", config_.machine_token);
        tc.reconnect_interval = j.value("reconnect_interval", 5);

        if (tunnel_client_->configure(tc)) {
            // Update kernel config
            config_.relay_url = tc.relay_url;
            config_.machine_id = tc.machine_id;
            config_.machine_token = tc.token;

            json response;
            response["success"] = true;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_TUNNEL_CONFIG, response.dump());
        } else {
            json response;
            response["success"] = false;
            response["error"] = "Failed to configure tunnel";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_TUNNEL_CONFIG, response.dump());
        }

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_TUNNEL_CONFIG, response.dump());
    }
}

void Kernel::process_tunnel_events() {
    auto events = tunnel_client_->poll_events();

    for (const auto& event : events) {
        switch (event.type) {
            case TunnelEvent::Type::SYSCALL:
                // Process syscall from remote agent
                handle_tunnel_syscall(event.agent_id, event.opcode, event.payload);
                break;

            case TunnelEvent::Type::AGENT_CONNECTED:
                // Remote agent connected - could emit kernel event
                spdlog::info("Remote agent connected: {} (id={})",
                            event.agent_name, event.agent_id);
                break;

            case TunnelEvent::Type::AGENT_DISCONNECTED:
                spdlog::info("Remote agent disconnected: id={}", event.agent_id);
                break;

            case TunnelEvent::Type::DISCONNECTED:
                spdlog::warn("Tunnel disconnected from relay");
                break;

            case TunnelEvent::Type::RECONNECTED:
                spdlog::info("Tunnel reconnected to relay");
                break;

            case TunnelEvent::Type::ERROR:
                spdlog::error("Tunnel error: {}", event.error);
                break;
        }
    }
}

void Kernel::handle_tunnel_syscall(uint32_t agent_id, uint8_t opcode,
                                  const std::vector<uint8_t>& payload) {
    // Create a message as if it came from a local agent
    ipc::Message msg;
    msg.agent_id = agent_id;
    msg.opcode = static_cast<ipc::SyscallOp>(opcode);
    msg.payload = payload;

    spdlog::debug("Processing syscall from remote agent {}: opcode=0x{:02x}",
                  agent_id, opcode);

    // Process the message
    auto response = handle_message(msg);

    // Send response back through tunnel
    tunnel_client_->send_response(
        agent_id,
        static_cast<uint8_t>(response.opcode),
        response.payload
    );
}

// ============================================================================
// Metrics Syscall Handlers
// ============================================================================

ipc::Message Kernel::handle_metrics_system(const ipc::Message& msg) {
    // Collect system-wide metrics
    auto metrics = metrics_collector_->collect_system();

    json response;
    response["success"] = true;
    response["metrics"] = metrics.to_json();

    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_METRICS_SYSTEM, response.dump());
}

ipc::Message Kernel::handle_metrics_agent(const ipc::Message& msg) {
    json request;
    try {
        request = json::parse(msg.payload_str());
    } catch (...) {
        json response;
        response["success"] = false;
        response["error"] = "Invalid JSON payload";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_METRICS_AGENT, response.dump());
    }

    // Get agent ID from request, default to caller's ID
    uint32_t target_agent_id = request.value("agent_id", msg.agent_id);

    // Find the agent
    auto target_agent = agent_manager_->get_agent(target_agent_id);

    if (!target_agent) {
        json response;
        response["success"] = false;
        response["error"] = "Agent not found";
        response["agent_id"] = target_agent_id;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_METRICS_AGENT, response.dump());
    }

    // Get agent metrics (includes uptime calculation)
    auto agent_metrics = target_agent->get_metrics();

    // Determine cgroup path (sandboxed agents have cgroups)
    std::string cgroup_path;
    if (target_agent->is_running()) {
        cgroup_path = "clove/agent-" + std::to_string(target_agent->id());
    }

    // Collect detailed metrics
    auto metrics = metrics_collector_->collect_agent(
        target_agent->id(),
        target_agent->pid(),
        cgroup_path,
        target_agent->name(),
        runtime::agent_state_to_string(target_agent->state()),
        agent_metrics.uptime_seconds * 1000
    );

    json response;
    response["success"] = true;
    response["metrics"] = metrics.to_json();

    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_METRICS_AGENT, response.dump());
}

ipc::Message Kernel::handle_metrics_all_agents(const ipc::Message& msg) {
    auto agents = agent_manager_->list_agents();

    json agent_metrics_list = json::array();

    for (const auto& agent : agents) {
        // Get agent's internal metrics
        auto agent_info = agent->get_metrics();

        // Determine cgroup path
        std::string cgroup_path;
        if (agent->is_running()) {
            cgroup_path = "clove/agent-" + std::to_string(agent->id());
        }

        // Collect detailed metrics
        auto metrics = metrics_collector_->collect_agent(
            agent->id(),
            agent->pid(),
            cgroup_path,
            agent->name(),
            runtime::agent_state_to_string(agent->state()),
            agent_info.uptime_seconds * 1000
        );

        agent_metrics_list.push_back(metrics.to_json());
    }

    json response;
    response["success"] = true;
    response["agents"] = agent_metrics_list;
    response["count"] = agent_metrics_list.size();

    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_METRICS_ALL_AGENTS, response.dump());
}

ipc::Message Kernel::handle_metrics_cgroup(const ipc::Message& msg) {
    json request;
    try {
        request = json::parse(msg.payload_str());
    } catch (...) {
        json response;
        response["success"] = false;
        response["error"] = "Invalid JSON payload";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_METRICS_CGROUP, response.dump());
    }

    std::string cgroup_path = request.value("cgroup_path", "");

    // If no path specified, try to get caller's cgroup
    if (cgroup_path.empty()) {
        // Default to agent's cgroup if sandboxed
        cgroup_path = "clove/agent-" + std::to_string(msg.agent_id);
    }

    auto metrics = metrics_collector_->collect_cgroup(cgroup_path);

    json response;
    response["success"] = metrics.valid;
    if (metrics.valid) {
        response["metrics"] = metrics.to_json();
    } else {
        response["error"] = "Cgroup not found or not readable";
        response["cgroup_path"] = cgroup_path;
    }

    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_METRICS_CGROUP, response.dump());
}

} // namespace agentos::kernel
