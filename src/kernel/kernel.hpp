/**
 * Clove Kernel
 *
 * Main kernel class that orchestrates all subsystems:
 * - Reactor (epoll event loop)
 * - SocketServer (Unix domain socket IPC)
 * - AgentManager (process lifecycle)
 * - Permissions (access control)
 */
#pragma once
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include "kernel/config.hpp"
#include "ipc/protocol.hpp"
#include "services/llm/client.hpp"

namespace clove::ipc {
class SocketServer;
} // namespace clove::ipc

namespace clove::runtime {
class AgentManager;
} // namespace clove::runtime

namespace clove::services::tunnel {
class TunnelClient;
struct TunnelConfig;
} // namespace clove::services::tunnel

namespace clove::metrics {
class MetricsCollector;
} // namespace clove::metrics

namespace clove::worlds {
class WorldEngine;
} // namespace clove::worlds

namespace clove::kernel {

class AsyncTaskManager;
class AuditLogger;
class ExecutionLogger;
class EventBus;
class KernelContext;
class KernelModule;
class PermissionsStore;
class Reactor;
class StateStore;
class SyscallRouter;
class AgentMailboxRegistry;
class LlmQueue;

class Kernel {
public:
    using Config = KernelConfig;

    struct Dependencies {
        std::unique_ptr<Reactor> reactor;
        std::unique_ptr<ipc::SocketServer> socket_server;
        std::unique_ptr<runtime::AgentManager> agent_manager;
        std::unique_ptr<worlds::WorldEngine> world_engine;
        std::unique_ptr<services::tunnel::TunnelClient> tunnel_client;
        std::unique_ptr<services::llm::LLMClient> llm_client;
        std::unique_ptr<LlmQueue> llm_queue;
        std::unique_ptr<metrics::MetricsCollector> metrics_collector;
        std::unique_ptr<AuditLogger> audit_logger;
        std::unique_ptr<ExecutionLogger> execution_logger;
        std::unique_ptr<StateStore> state_store;
        std::unique_ptr<EventBus> event_bus;
        std::unique_ptr<AgentMailboxRegistry> mailbox_registry;
        std::unique_ptr<PermissionsStore> permissions_store;
        std::unique_ptr<AsyncTaskManager> async_tasks;
    };

    Kernel();
    explicit Kernel(const Config& config);
    Kernel(const Config& config, Dependencies deps);
    ~Kernel();

    // Non-copyable
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    // Initialize all subsystems
    bool init();

    // Run the kernel (blocks until shutdown)
    void run();

    // Request shutdown
    void shutdown();

    // Check if running
    bool is_running() const { return running_; }

    // Access to agent manager
    runtime::AgentManager& agents();

    // Get config
    const Config& get_config() const { return config_; }

private:
    Config config_;
    std::atomic<bool> running_{false};

    std::unique_ptr<Reactor> reactor_;
    std::unique_ptr<ipc::SocketServer> socket_server_;
    std::unique_ptr<runtime::AgentManager> agent_manager_;
    std::unique_ptr<worlds::WorldEngine> world_engine_;
    std::unique_ptr<services::tunnel::TunnelClient> tunnel_client_;
    std::unique_ptr<services::llm::LLMClient> llm_client_;
    std::unique_ptr<LlmQueue> llm_queue_;
    std::unique_ptr<metrics::MetricsCollector> metrics_collector_;
    std::unique_ptr<AuditLogger> audit_logger_;
    std::unique_ptr<ExecutionLogger> execution_logger_;
    std::unique_ptr<SyscallRouter> syscall_router_;
    std::unique_ptr<StateStore> state_store_;
    std::unique_ptr<EventBus> event_bus_;
    std::unique_ptr<AgentMailboxRegistry> mailbox_registry_;
    std::unique_ptr<PermissionsStore> permissions_store_;
    std::unique_ptr<AsyncTaskManager> async_tasks_;
    std::unique_ptr<KernelContext> context_;
    std::vector<std::unique_ptr<KernelModule>> modules_;

    // Event handlers
    void on_server_event(int fd, uint32_t events);
    void on_client_event(int fd, uint32_t events);

    // Message handler
    ipc::Message handle_message(const ipc::Message& msg);

    // Update client in reactor (for write events)
    void update_client_events(int fd);
};

} // namespace clove::kernel
