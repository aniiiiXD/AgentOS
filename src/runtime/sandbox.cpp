#include "runtime/sandbox.hpp"
#include <spdlog/spdlog.h>

#include <sys/wait.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace agentos::runtime {

// Stack size for clone()
constexpr size_t STACK_SIZE = 1024 * 1024; // 1MB

// Child process arguments
struct ChildArgs {
    Sandbox* sandbox;
    std::string command;
    std::vector<std::string> args;
    int pipe_fd[2]; // For synchronization
};

// ============================================================================
// Sandbox Implementation
// ============================================================================

Sandbox::Sandbox(const SandboxConfig& config)
    : config_(config) {
    cgroup_path_ = "/sys/fs/cgroup/agentos/" + config_.name;
}

Sandbox::~Sandbox() {
    if (state_ == SandboxState::RUNNING) {
        stop();
    }
    destroy();
}

bool Sandbox::create() {
    if (state_ != SandboxState::CREATED) {
        spdlog::error("Sandbox {} already created", config_.name);
        return false;
    }

    spdlog::info("Creating sandbox: {}", config_.name);

    // Setup cgroups if enabled
    if (config_.enable_cgroups) {
        if (!setup_cgroups()) {
            spdlog::error("Failed to setup cgroups for {}", config_.name);
            set_state(SandboxState::FAILED);
            return false;
        }
    }

    spdlog::debug("Sandbox {} created successfully", config_.name);
    return true;
}

bool Sandbox::setup_cgroups() {
    // Check if cgroup v2 is available
    if (!fs::exists("/sys/fs/cgroup/cgroup.controllers")) {
        spdlog::warn("cgroup v2 not available, skipping cgroup setup");
        return true;
    }

    // Create agentos cgroup directory if needed
    std::string agentos_cgroup = "/sys/fs/cgroup/agentos";
    if (!fs::exists(agentos_cgroup)) {
        try {
            fs::create_directories(agentos_cgroup);
        } catch (const std::exception& e) {
            spdlog::warn("Cannot create cgroup dir (need root): {}", e.what());
            return true; // Continue without cgroups
        }
    }

    // Create sandbox-specific cgroup
    if (!fs::exists(cgroup_path_)) {
        try {
            fs::create_directories(cgroup_path_);
        } catch (const std::exception& e) {
            spdlog::warn("Cannot create sandbox cgroup (need root): {}", e.what());
            return true;
        }
    }

    // Set memory limit
    std::string memory_max = cgroup_path_ + "/memory.max";
    if (fs::exists(cgroup_path_ + "/memory.max")) {
        std::ofstream(memory_max) << config_.limits.memory_limit_bytes;
        spdlog::debug("Set memory limit: {} bytes", config_.limits.memory_limit_bytes);
    }

    // Set CPU quota
    std::string cpu_max = cgroup_path_ + "/cpu.max";
    if (fs::exists(cgroup_path_ + "/cpu.max")) {
        std::ofstream(cpu_max) << config_.limits.cpu_quota_us << " "
                               << config_.limits.cpu_period_us;
        spdlog::debug("Set CPU quota: {}us per {}us",
            config_.limits.cpu_quota_us, config_.limits.cpu_period_us);
    }

    // Set max PIDs
    std::string pids_max = cgroup_path_ + "/pids.max";
    if (fs::exists(cgroup_path_ + "/pids.max")) {
        std::ofstream(pids_max) << config_.limits.max_pids;
        spdlog::debug("Set max PIDs: {}", config_.limits.max_pids);
    }

    return true;
}

bool Sandbox::cleanup_cgroups() {
    if (fs::exists(cgroup_path_)) {
        try {
            fs::remove_all(cgroup_path_);
            spdlog::debug("Cleaned up cgroup: {}", cgroup_path_);
        } catch (const std::exception& e) {
            spdlog::warn("Failed to cleanup cgroup: {}", e.what());
        }
    }
    return true;
}

int Sandbox::child_entry(void* arg) {
    ChildArgs* child_args = static_cast<ChildArgs*>(arg);
    Sandbox* sandbox = child_args->sandbox;

    // Close write end of pipe
    close(child_args->pipe_fd[1]);

    // Wait for parent to set up cgroups
    char buf;
    read(child_args->pipe_fd[0], &buf, 1);
    close(child_args->pipe_fd[0]);

    // Set hostname if UTS namespace is enabled
    if (sandbox->config_.enable_uts_namespace) {
        std::string hostname = "agentos-" + sandbox->config_.name;
        if (sethostname(hostname.c_str(), hostname.length()) < 0) {
            spdlog::warn("Failed to set hostname: {}", strerror(errno));
        }
    }

    // Mount proc if PID namespace is enabled
    if (sandbox->config_.enable_pid_namespace && sandbox->config_.enable_mount_namespace) {
        // Create a new proc mount point
        if (mount("proc", "/proc", "proc", 0, nullptr) < 0) {
            // This might fail without root, that's ok
            spdlog::debug("Could not mount /proc (may need root)");
        }
    }

    // Build argv for execvp
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(child_args->command.c_str()));
    for (auto& arg : child_args->args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    // Execute the command
    execvp(argv[0], argv.data());

    // If we get here, exec failed
    spdlog::error("execvp failed: {}", strerror(errno));
    _exit(127);
}

bool Sandbox::start(const std::string& command, const std::vector<std::string>& args) {
    if (state_ == SandboxState::RUNNING) {
        spdlog::error("Sandbox {} already running", config_.name);
        return false;
    }

    spdlog::info("Starting sandbox {} with command: {}", config_.name, command);

    // Create synchronization pipe
    int pipe_fd[2];
    if (pipe(pipe_fd) < 0) {
        spdlog::error("Failed to create pipe: {}", strerror(errno));
        set_state(SandboxState::FAILED);
        return false;
    }

    // Prepare child arguments
    ChildArgs child_args;
    child_args.sandbox = this;
    child_args.command = command;
    child_args.args = args;
    child_args.pipe_fd[0] = pipe_fd[0];
    child_args.pipe_fd[1] = pipe_fd[1];

    // Build clone flags
    int clone_flags = SIGCHLD;

    if (config_.enable_pid_namespace) {
        clone_flags |= CLONE_NEWPID;
    }
    if (config_.enable_mount_namespace) {
        clone_flags |= CLONE_NEWNS;
    }
    if (config_.enable_uts_namespace) {
        clone_flags |= CLONE_NEWUTS;
    }
    if (!config_.enable_network) {
        clone_flags |= CLONE_NEWNET; // Isolate network
    }

    // Allocate stack for child
    char* stack = new char[STACK_SIZE];
    char* stack_top = stack + STACK_SIZE;

    // Clone the process
    child_pid_ = clone(child_entry, stack_top, clone_flags, &child_args);

    if (child_pid_ < 0) {
        spdlog::error("clone() failed: {}", strerror(errno));
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        delete[] stack;

        // Try fallback to fork if clone fails (no root)
        spdlog::info("Falling back to fork() (no namespace isolation)");
        child_pid_ = fork();

        if (child_pid_ < 0) {
            spdlog::error("fork() failed: {}", strerror(errno));
            set_state(SandboxState::FAILED);
            return false;
        }

        if (child_pid_ == 0) {
            // Child process
            std::vector<char*> argv;
            argv.push_back(const_cast<char*>(command.c_str()));
            for (const auto& arg : args) {
                argv.push_back(const_cast<char*>(arg.c_str()));
            }
            argv.push_back(nullptr);

            execvp(argv[0], argv.data());
            _exit(127);
        }

        set_state(SandboxState::RUNNING);
        spdlog::info("Sandbox {} started (PID={}, no isolation)", config_.name, child_pid_);
        return true;
    }

    // Parent: close read end
    close(pipe_fd[0]);

    // Add child to cgroup
    if (config_.enable_cgroups && fs::exists(cgroup_path_ + "/cgroup.procs")) {
        std::ofstream(cgroup_path_ + "/cgroup.procs") << child_pid_;
        spdlog::debug("Added PID {} to cgroup {}", child_pid_, cgroup_path_);
    }

    // Signal child to continue
    write(pipe_fd[1], "x", 1);
    close(pipe_fd[1]);

    delete[] stack;

    set_state(SandboxState::RUNNING);
    spdlog::info("Sandbox {} started (PID={})", config_.name, child_pid_);

    return true;
}

bool Sandbox::stop(int timeout_ms) {
    if (state_ != SandboxState::RUNNING) {
        return true;
    }

    spdlog::info("Stopping sandbox {} (PID={})", config_.name, child_pid_);

    // First try SIGTERM
    if (kill(child_pid_, SIGTERM) < 0) {
        if (errno == ESRCH) {
            // Process already dead
            set_state(SandboxState::STOPPED);
            return true;
        }
        spdlog::error("kill(SIGTERM) failed: {}", strerror(errno));
    }

    // Wait for process to exit with timeout
    int waited = 0;
    int interval = 100; // 100ms

    while (waited < timeout_ms) {
        int status;
        pid_t result = waitpid(child_pid_, &status, WNOHANG);

        if (result == child_pid_) {
            exit_code_ = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            set_state(SandboxState::STOPPED);
            spdlog::info("Sandbox {} stopped (exit={})", config_.name, exit_code_);
            return true;
        }

        if (result < 0 && errno != EINTR) {
            break;
        }

        usleep(interval * 1000);
        waited += interval;
    }

    // Force kill
    spdlog::warn("Sandbox {} not responding, sending SIGKILL", config_.name);
    kill(child_pid_, SIGKILL);
    waitpid(child_pid_, nullptr, 0);

    set_state(SandboxState::STOPPED);
    return true;
}

bool Sandbox::destroy() {
    if (state_ == SandboxState::RUNNING) {
        stop();
    }

    cleanup_cgroups();
    spdlog::debug("Sandbox {} destroyed", config_.name);
    return true;
}

int Sandbox::wait() {
    if (child_pid_ <= 0) {
        return -1;
    }

    int status;
    waitpid(child_pid_, &status, 0);

    if (WIFEXITED(status)) {
        exit_code_ = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        exit_code_ = 128 + WTERMSIG(status);
    }

    set_state(SandboxState::STOPPED);
    return exit_code_;
}

bool Sandbox::is_running() const {
    if (child_pid_ <= 0 || state_ != SandboxState::RUNNING) {
        return false;
    }

    // Check if process is still alive
    if (kill(child_pid_, 0) < 0) {
        return false;
    }

    return true;
}

void Sandbox::set_event_callback(SandboxEventCallback callback) {
    event_callback_ = std::move(callback);
}

void Sandbox::set_state(SandboxState new_state) {
    state_ = new_state;
    if (event_callback_) {
        event_callback_(this, new_state);
    }
}

// ============================================================================
// SandboxManager Implementation
// ============================================================================

SandboxManager::SandboxManager() {
    cgroup_root_ = "/sys/fs/cgroup/agentos";
    init_cgroup_root();
}

SandboxManager::~SandboxManager() {
    cleanup_all();
}

bool SandboxManager::is_available() {
    // Check for clone() with namespaces (needs CAP_SYS_ADMIN or root)
    // For now, we'll always return true and fallback to fork() if needed
    return true;
}

bool SandboxManager::init_cgroup_root() {
    if (!fs::exists("/sys/fs/cgroup/cgroup.controllers")) {
        spdlog::warn("cgroup v2 not available");
        return false;
    }

    if (!fs::exists(cgroup_root_)) {
        try {
            fs::create_directories(cgroup_root_);
            spdlog::info("Created cgroup root: {}", cgroup_root_);
        } catch (const std::exception& e) {
            spdlog::warn("Cannot create cgroup root (need root): {}", e.what());
            return false;
        }
    }

    // Enable controllers
    std::string subtree_control = "/sys/fs/cgroup/cgroup.subtree_control";
    if (fs::exists(subtree_control)) {
        try {
            std::ofstream(subtree_control) << "+cpu +memory +pids";
        } catch (...) {
            spdlog::debug("Could not enable cgroup controllers");
        }
    }

    return true;
}

std::shared_ptr<Sandbox> SandboxManager::create_sandbox(const SandboxConfig& config) {
    if (sandboxes_.count(config.name)) {
        spdlog::error("Sandbox {} already exists", config.name);
        return nullptr;
    }

    auto sandbox = std::make_shared<Sandbox>(config);
    if (!sandbox->create()) {
        return nullptr;
    }

    sandboxes_[config.name] = sandbox;
    return sandbox;
}

std::shared_ptr<Sandbox> SandboxManager::get_sandbox(const std::string& name) {
    auto it = sandboxes_.find(name);
    if (it != sandboxes_.end()) {
        return it->second;
    }
    return nullptr;
}

bool SandboxManager::remove_sandbox(const std::string& name) {
    auto it = sandboxes_.find(name);
    if (it == sandboxes_.end()) {
        return false;
    }

    it->second->destroy();
    sandboxes_.erase(it);
    return true;
}

std::vector<std::string> SandboxManager::list_sandboxes() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : sandboxes_) {
        names.push_back(name);
    }
    return names;
}

void SandboxManager::cleanup_all() {
    for (auto& [name, sandbox] : sandboxes_) {
        sandbox->destroy();
    }
    sandboxes_.clear();

    // Cleanup cgroup root
    if (fs::exists(cgroup_root_)) {
        try {
            fs::remove_all(cgroup_root_);
        } catch (...) {}
    }
}

} // namespace agentos::runtime
