#include "kernel/kernel.hpp"
#include "kernel/context.hpp"
#include "kernel/syscall_router.hpp"
#include "kernel/syscall_handlers.hpp"
#include "kernel/async_task_manager.hpp"
#include "kernel/audit_log.hpp"
#include "kernel/event_bus.hpp"
#include "kernel/execution_log.hpp"
#include "kernel/ipc_mailbox.hpp"
#include "kernel/llm_queue.hpp"
#include "kernel/permissions_store.hpp"
#include "kernel/reactor.hpp"
#include "kernel/state_store.hpp"
#include "ipc/transport/socket_server.hpp"
#include "metrics/metrics.hpp"
#include "runtime/agent/manager.hpp"
#include "services/llm/client.hpp"
#include "services/tunnel/client.hpp"
#include "worlds/world_engine.hpp"
#include <spdlog/spdlog.h>
#include <sys/epoll.h>
#include <sys/wait.h>
#include <csignal>
#include <fstream>
#include <thread>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace clove::kernel {

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
    : Kernel(config, Dependencies{}) {}

Kernel::Kernel(const Config& config, Dependencies deps)
    : config_(config)
{
    reactor_ = std::move(deps.reactor);
    socket_server_ = std::move(deps.socket_server);
    agent_manager_ = std::move(deps.agent_manager);
    world_engine_ = std::move(deps.world_engine);
    tunnel_client_ = std::move(deps.tunnel_client);
    llm_client_ = std::move(deps.llm_client);
    llm_queue_ = std::move(deps.llm_queue);
    metrics_collector_ = std::move(deps.metrics_collector);
    audit_logger_ = std::move(deps.audit_logger);
    execution_logger_ = std::move(deps.execution_logger);
    state_store_ = std::move(deps.state_store);
    event_bus_ = std::move(deps.event_bus);
    mailbox_registry_ = std::move(deps.mailbox_registry);
    permissions_store_ = std::move(deps.permissions_store);
    async_tasks_ = std::move(deps.async_tasks);

    if (!reactor_) {
        reactor_ = std::make_unique<Reactor>();
    }
    if (!socket_server_) {
        socket_server_ = std::make_unique<ipc::SocketServer>(config.socket_path);
    }
    if (!agent_manager_) {
        agent_manager_ = std::make_unique<runtime::AgentManager>(config.socket_path);
    }
    if (!world_engine_) {
        world_engine_ = std::make_unique<clove::worlds::WorldEngine>();
    }
    if (!tunnel_client_) {
        tunnel_client_ = std::make_unique<clove::services::tunnel::TunnelClient>();
    }
    if (!llm_client_) {
        services::llm::LLMConfig llm_config;
        llm_config.api_key = config.gemini_api_key;
        llm_config.model = config.llm_model;
        llm_client_ = std::make_unique<services::llm::LLMClient>(llm_config);
    }
    if (!llm_queue_) {
        llm_queue_ = std::make_unique<LlmQueue>(*llm_client_);
    }
    if (!metrics_collector_) {
        metrics_collector_ = std::make_unique<clove::metrics::MetricsCollector>();
    }
    if (!audit_logger_) {
        audit_logger_ = std::make_unique<AuditLogger>();
    }
    if (!execution_logger_) {
        execution_logger_ = std::make_unique<ExecutionLogger>();
    }
    if (!state_store_) {
        state_store_ = std::make_unique<StateStore>();
    }
    if (!event_bus_) {
        event_bus_ = std::make_unique<EventBus>();
    }
    if (!mailbox_registry_) {
        mailbox_registry_ = std::make_unique<AgentMailboxRegistry>();
    }
    if (!permissions_store_) {
        permissions_store_ = std::make_unique<PermissionsStore>();
    }
    if (!async_tasks_) {
        async_tasks_ = std::make_unique<AsyncTaskManager>();
    }

    context_ = std::make_unique<KernelContext>(KernelContext{
        config_,
        *reactor_,
        *socket_server_,
        *agent_manager_,
        *world_engine_,
        *tunnel_client_,
        *metrics_collector_,
        *audit_logger_,
        *execution_logger_,
        *state_store_,
        *event_bus_,
        *mailbox_registry_,
        *permissions_store_,
        *async_tasks_,
        *llm_queue_
    });

    syscall_router_ = std::make_unique<SyscallRouter>();
    modules_.push_back(std::make_unique<AgentSyscalls>(*context_));
    modules_.push_back(std::make_unique<AsyncSyscalls>(*context_));
    modules_.push_back(std::make_unique<AuditSyscalls>(*context_));
    modules_.push_back(std::make_unique<EventSyscalls>(*context_));
    modules_.push_back(std::make_unique<ExecSyscalls>(*context_));
    modules_.push_back(std::make_unique<FileSyscalls>(*context_));
    modules_.push_back(std::make_unique<IpcSyscalls>(*context_));
    modules_.push_back(std::make_unique<LlmSyscalls>(*context_));
    modules_.push_back(std::make_unique<MetricsSyscalls>(*context_));
    modules_.push_back(std::make_unique<NetworkSyscalls>(*context_));
    modules_.push_back(std::make_unique<PermissionSyscalls>(*context_));
    modules_.push_back(std::make_unique<ReplaySyscalls>(*context_));
    modules_.push_back(std::make_unique<StateSyscalls>(*context_));
    modules_.push_back(std::make_unique<TunnelSyscalls>(*context_,
        [this](const ipc::Message& msg) { return syscall_router_->handle(msg); }));
    modules_.push_back(std::make_unique<WorldSyscalls>(*context_));

    syscall_router_->register_handler(ipc::SyscallOp::SYS_NOOP,
        [](const ipc::Message& msg) {
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_NOOP, msg.payload);
        });

    syscall_router_->register_handler(ipc::SyscallOp::SYS_EXIT,
        [](const ipc::Message& msg) {
            spdlog::info("Agent {} requested exit", msg.agent_id);
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EXIT, "goodbye");
        });

    syscall_router_->register_handler(ipc::SyscallOp::SYS_HELLO,
        [](const ipc::Message& msg) {
            json response;
            response["success"] = true;
            response["protocol_version"] = ipc::PROTOCOL_VERSION;
            response["kernel_version"] = "0.1.0";
            response["features"] = {
                {"llm_in_kernel", false},
                {"sys_think_stub", true},
                {"async_default_exec_http", true}
            };
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HELLO, response.dump());
        });

    for (auto& module : modules_) {
        module->register_syscalls(*syscall_router_);
    }
}

Kernel::~Kernel() {
    if (g_kernel == this) {
        g_kernel = nullptr;
    }
}

bool Kernel::init() {
    spdlog::info("Initializing Clove Kernel...");

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
                event_bus_->emit(KernelEventType::AGENT_RESTARTING, event_data, 0);
            } else if (event_type == "AGENT_ESCALATED") {
                event_bus_->emit(KernelEventType::AGENT_ESCALATED, event_data, 0);
            }
        });

    // Initialize tunnel client
    if (tunnel_client_->init()) {
        spdlog::info("Tunnel client initialized");

        // Configure if relay URL is set
        if (!config_.relay_url.empty()) {
            services::tunnel::TunnelConfig tc;
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
    spdlog::info("Clove Kernel v0.1.0 running");
    spdlog::info("Listening on: {}", config_.socket_path);
    spdlog::info("Press Ctrl+C to exit");

    while (running_) {
        int n = reactor_->poll(100);
        if (n < 0) {
            spdlog::error("Reactor error, exiting");
            break;
        }

        // Allow modules to run periodic work (e.g. tunnel event pump)
        for (auto& module : modules_) {
            module->on_tick();
        }

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

runtime::AgentManager& Kernel::agents() {
    return *agent_manager_;
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
        uint32_t agent_id = socket_server_->remove_client(fd);
        if (agent_id > 0) {
            context_->mailbox_registry.unregister(agent_id);
        }
        return;
    }

    // Handle readable
    if (events & EPOLLIN) {
        if (!socket_server_->handle_client(fd)) {
            reactor_->remove(fd);
            uint32_t agent_id = socket_server_->remove_client(fd);
            if (agent_id > 0) {
                context_->mailbox_registry.unregister(agent_id);
            }
            return;
        }
    }

    // Handle writable
    if (events & EPOLLOUT) {
        if (!socket_server_->flush_client(fd)) {
            reactor_->remove(fd);
            uint32_t agent_id = socket_server_->remove_client(fd);
            if (agent_id > 0) {
                context_->mailbox_registry.unregister(agent_id);
            }
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
    return syscall_router_->handle(msg);
}

// ============================================================================
// IPC Handlers - Inter-Agent Communication
// ============================================================================

// ============================================================================
// Audit Syscall Handlers
// ============================================================================

// ============================================================================
// Replay Syscall Handlers
// ============================================================================

} // namespace clove::kernel
