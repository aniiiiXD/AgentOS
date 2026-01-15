#pragma once
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>
#include <sys/types.h>

namespace agentos::runtime {

// Resource limits for sandboxed processes
struct ResourceLimits {
    uint64_t memory_limit_bytes = 256 * 1024 * 1024;  // 256MB default
    uint64_t cpu_shares = 1024;                        // Relative CPU weight
    uint64_t cpu_quota_us = 100000;                    // 100ms per 100ms period (100%)
    uint64_t cpu_period_us = 100000;                   // 100ms period
    uint64_t max_pids = 64;                            // Max processes
};

// Sandbox configuration
struct SandboxConfig {
    std::string name;                    // Unique sandbox name
    std::string root_path = "/tmp";      // Root filesystem path
    std::string socket_path;             // Socket for IPC with kernel
    ResourceLimits limits;

    bool enable_network = false;         // Network isolation
    bool enable_pid_namespace = true;    // PID namespace isolation
    bool enable_mount_namespace = true;  // Mount namespace isolation
    bool enable_uts_namespace = true;    // UTS (hostname) isolation
    bool enable_cgroups = true;          // cgroups resource limits
};

// Sandbox state
enum class SandboxState {
    CREATED,
    RUNNING,
    STOPPED,
    FAILED
};

// Forward declaration
class Sandbox;

// Callback for sandbox events
using SandboxEventCallback = std::function<void(Sandbox*, SandboxState)>;

class Sandbox {
public:
    explicit Sandbox(const SandboxConfig& config);
    ~Sandbox();

    // Non-copyable
    Sandbox(const Sandbox&) = delete;
    Sandbox& operator=(const Sandbox&) = delete;

    // Lifecycle
    bool create();
    bool start(const std::string& command, const std::vector<std::string>& args = {});
    bool stop(int timeout_ms = 5000);
    bool destroy();

    // Status
    SandboxState state() const { return state_; }
    pid_t pid() const { return child_pid_; }
    const std::string& name() const { return config_.name; }
    const SandboxConfig& config() const { return config_; }

    // Wait for sandbox to exit
    int wait();

    // Check if process is still running
    bool is_running() const;

    // Get exit code (only valid after wait())
    int exit_code() const { return exit_code_; }

    // Event callback
    void set_event_callback(SandboxEventCallback callback);

private:
    SandboxConfig config_;
    SandboxState state_ = SandboxState::CREATED;
    pid_t child_pid_ = -1;
    int exit_code_ = -1;
    SandboxEventCallback event_callback_;

    // cgroup paths
    std::string cgroup_path_;

    // Internal methods
    bool setup_cgroups();
    bool cleanup_cgroups();
    bool setup_namespaces();

    // Child process entry point (runs in new namespaces)
    static int child_entry(void* arg);

    void set_state(SandboxState new_state);
};

// Sandbox manager - handles multiple sandboxes
class SandboxManager {
public:
    SandboxManager();
    ~SandboxManager();

    // Check if sandboxing is available on this system
    static bool is_available();

    // Create a new sandbox
    std::shared_ptr<Sandbox> create_sandbox(const SandboxConfig& config);

    // Get sandbox by name
    std::shared_ptr<Sandbox> get_sandbox(const std::string& name);

    // Remove sandbox
    bool remove_sandbox(const std::string& name);

    // List all sandboxes
    std::vector<std::string> list_sandboxes() const;

    // Cleanup all sandboxes
    void cleanup_all();

private:
    std::unordered_map<std::string, std::shared_ptr<Sandbox>> sandboxes_;
    std::string cgroup_root_;

    bool init_cgroup_root();
};

} // namespace agentos::runtime
