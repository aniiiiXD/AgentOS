#pragma once
#include <string>
#include <memory>
#include <atomic>
#include "kernel/reactor.hpp"
#include "kernel/llm_client.hpp"
#include "ipc/socket_server.hpp"
#include "runtime/agent_process.hpp"

namespace agentos::kernel {

// Kernel configuration
struct KernelConfig {
    std::string socket_path = "/tmp/agentos.sock";
    bool enable_sandboxing = true;
    std::string gemini_api_key;          // Gemini API key (or from env)
    std::string llm_model = "gemini-2.0-flash";
};

class Kernel {
public:
    using Config = KernelConfig;

    Kernel();
    explicit Kernel(const Config& config);
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
    runtime::AgentManager& agents() { return *agent_manager_; }

private:
    Config config_;
    std::atomic<bool> running_{false};

    std::unique_ptr<Reactor> reactor_;
    std::unique_ptr<ipc::SocketServer> socket_server_;
    std::unique_ptr<runtime::AgentManager> agent_manager_;
    std::unique_ptr<LLMClient> llm_client_;

    // Event handlers
    void on_server_event(int fd, uint32_t events);
    void on_client_event(int fd, uint32_t events);

    // Message handler
    ipc::Message handle_message(const ipc::Message& msg);

    // Syscall handlers
    ipc::Message handle_think(const ipc::Message& msg);
    ipc::Message handle_spawn(const ipc::Message& msg);
    ipc::Message handle_kill(const ipc::Message& msg);
    ipc::Message handle_list(const ipc::Message& msg);

    // Update client in reactor (for write events)
    void update_client_events(int fd);
};

} // namespace agentos::kernel
