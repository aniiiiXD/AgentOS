#include "runtime/agent/process.hpp"
#include <spdlog/spdlog.h>
#include <atomic>
#include <chrono>
#include <fstream>

namespace clove::runtime {

// ============================================================================
// AgentProcess Implementation
// ============================================================================

static std::atomic<uint32_t> g_next_agent_id{1};

uint32_t AgentProcess::generate_id() {
    return g_next_agent_id++;
}

AgentProcess::AgentProcess(const AgentConfig& config)
    : config_(config)
    , id_(generate_id()) {
    // Record creation timestamp
    auto now = std::chrono::system_clock::now();
    created_at_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    spdlog::debug("AgentProcess created: {} (id={})", config_.name, id_);
}

AgentProcess::~AgentProcess() {
    if (state_ == AgentState::RUNNING || state_ == AgentState::STARTING || state_ == AgentState::PAUSED) {
        stop();
    }
}

bool AgentProcess::start() {
    if (state_ == AgentState::RUNNING) {
        spdlog::warn("Agent {} already running", config_.name);
        return false;
    }

    set_state(AgentState::STARTING);
    spdlog::info("Starting agent: {} (id={})", config_.name, id_);

    // Create sandbox configuration
    SandboxConfig sandbox_config;
    sandbox_config.name = config_.name + "_" + std::to_string(id_);
    sandbox_config.socket_path = config_.socket_path;
    sandbox_config.limits = config_.limits;
    sandbox_config.enable_network = config_.enable_network;
    sandbox_config.enable_cgroups = config_.sandboxed;
    sandbox_config.enable_pid_namespace = config_.sandboxed;
    sandbox_config.enable_mount_namespace = config_.sandboxed;
    sandbox_config.enable_uts_namespace = config_.sandboxed;

    // Create sandbox
    sandbox_ = std::make_unique<Sandbox>(sandbox_config);

    if (!sandbox_->create()) {
        spdlog::error("Failed to create sandbox for agent {}", config_.name);
        set_state(AgentState::FAILED);
        return false;
    }

    // Build command arguments
    auto args = build_args();

    // Start the process
    if (!sandbox_->start(config_.python_path, args)) {
        spdlog::error("Failed to start agent {}", config_.name);
        set_state(AgentState::FAILED);
        return false;
    }

    set_state(AgentState::RUNNING);
    spdlog::info("Agent {} started (pid={})", config_.name, sandbox_->pid());

    return true;
}

bool AgentProcess::stop(int timeout_ms) {
    if (state_ != AgentState::RUNNING && state_ != AgentState::STARTING && state_ != AgentState::PAUSED) {
        return true;
    }

    set_state(AgentState::STOPPING);
    spdlog::info("Stopping agent: {} (id={})", config_.name, id_);

    if (sandbox_) {
        sandbox_->stop(timeout_ms);
        exit_code_ = sandbox_->exit_code();
    }

    set_state(AgentState::STOPPED);
    spdlog::info("Agent {} stopped (exit={})", config_.name, exit_code_);

    return true;
}

bool AgentProcess::restart() {
    stop();
    return start();
}

bool AgentProcess::pause() {
    if (state_ != AgentState::RUNNING) {
        spdlog::error("Cannot pause agent {} - not running (state={})",
                      config_.name, agent_state_to_string(state_));
        return false;
    }

    if (!sandbox_) {
        spdlog::error("Cannot pause agent {} - no sandbox", config_.name);
        return false;
    }

    spdlog::info("Pausing agent: {} (id={})", config_.name, id_);

    if (!sandbox_->pause()) {
        spdlog::error("Failed to pause sandbox for agent {}", config_.name);
        return false;
    }

    set_state(AgentState::PAUSED);
    spdlog::info("Agent {} paused", config_.name);
    return true;
}

bool AgentProcess::resume() {
    if (state_ != AgentState::PAUSED) {
        spdlog::error("Cannot resume agent {} - not paused (state={})",
                      config_.name, agent_state_to_string(state_));
        return false;
    }

    if (!sandbox_) {
        spdlog::error("Cannot resume agent {} - no sandbox", config_.name);
        return false;
    }

    spdlog::info("Resuming agent: {} (id={})", config_.name, id_);

    if (!sandbox_->resume()) {
        spdlog::error("Failed to resume sandbox for agent {}", config_.name);
        return false;
    }

    set_state(AgentState::RUNNING);
    spdlog::info("Agent {} resumed", config_.name);
    return true;
}

pid_t AgentProcess::pid() const {
    return sandbox_ ? sandbox_->pid() : -1;
}

int AgentProcess::wait() {
    if (!sandbox_) {
        return -1;
    }

    int code = sandbox_->wait();
    exit_code_ = code;
    set_state(AgentState::STOPPED);
    return code;
}

bool AgentProcess::is_running() const {
    if (state_ != AgentState::RUNNING) {
        return false;
    }
    return sandbox_ && sandbox_->is_running();
}

int AgentProcess::exit_code() const {
    // Fetch from sandbox if available (sandbox updates exit code when process exits)
    if (sandbox_) {
        return sandbox_->exit_code();
    }
    return exit_code_;
}

void AgentProcess::set_event_callback(AgentEventCallback callback) {
    event_callback_ = std::move(callback);
}

AgentMetrics AgentProcess::get_metrics() const {
    AgentMetrics metrics;
    metrics.id = id_;
    metrics.name = config_.name;
    metrics.pid = pid();
    metrics.state = state_;

    // Calculate uptime
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    metrics.uptime_seconds = (now_ms - created_at_ms_) / 1000;

    // Read memory usage from cgroups
    metrics.memory_bytes = 0;
    metrics.cpu_percent = 0.0;

    if (config_.sandboxed && sandbox_ && state_ == AgentState::RUNNING) {
        std::string cgroup_path = "/sys/fs/cgroup/clove/" + config_.name + "_" + std::to_string(id_);

        // Read memory.current
        std::ifstream mem_file(cgroup_path + "/memory.current");
        if (mem_file) {
            mem_file >> metrics.memory_bytes;
        }

        // CPU percentage calculation requires tracking over time
        // For now, we'll leave it at 0 (can be enhanced later)
    }

    // LLM metrics
    metrics.llm_request_count = llm_request_count_;
    metrics.llm_tokens_used = llm_tokens_used_;

    // Hierarchy
    metrics.parent_id = parent_id_;
    metrics.child_ids = child_ids_;

    // Timestamps
    metrics.created_at_ms = created_at_ms_;

    return metrics;
}

void AgentProcess::record_llm_call(int tokens) {
    llm_request_count_++;
    llm_tokens_used_ += tokens;
    spdlog::debug("Agent {} LLM call: {} tokens (total: {})",
        config_.name, tokens, llm_tokens_used_);
}

IsolationStatus AgentProcess::get_isolation_status() const {
    if (sandbox_) {
        return sandbox_->isolation_status();
    }

    // No sandbox - return a status indicating no isolation
    IsolationStatus status;
    status.fully_isolated = false;
    status.degraded_reason = config_.sandboxed ?
        "Sandbox not initialized" : "Sandboxing disabled";
    return status;
}

void AgentProcess::add_child(uint32_t child_id) {
    child_ids_.push_back(child_id);
    spdlog::debug("Agent {} added child: {}", config_.name, child_id);
}

void AgentProcess::set_state(AgentState new_state) {
    spdlog::debug("Agent {} state: {} -> {}",
        config_.name,
        agent_state_to_string(state_),
        agent_state_to_string(new_state));

    state_ = new_state;

    if (event_callback_) {
        event_callback_(this, new_state);
    }
}

std::vector<std::string> AgentProcess::build_args() const {
    std::vector<std::string> args;

    // Add script path
    args.push_back(config_.script_path);

    // Add socket path as argument
    if (!config_.socket_path.empty()) {
        args.push_back(config_.socket_path);
    }

    return args;
}

} // namespace clove::runtime
