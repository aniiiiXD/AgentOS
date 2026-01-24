#include "runtime/agent_process.hpp"
#include <spdlog/spdlog.h>
#include <atomic>
#include <chrono>
#include <fstream>

namespace agentos::runtime {

// ============================================================================
// Utility
// ============================================================================

const char* agent_state_to_string(AgentState state) {
    switch (state) {
        case AgentState::CREATED:  return "CREATED";
        case AgentState::STARTING: return "STARTING";
        case AgentState::RUNNING:  return "RUNNING";
        case AgentState::STOPPING: return "STOPPING";
        case AgentState::STOPPED:  return "STOPPED";
        case AgentState::FAILED:   return "FAILED";
        default: return "UNKNOWN";
    }
}

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
    if (state_ == AgentState::RUNNING || state_ == AgentState::STARTING) {
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
    if (state_ != AgentState::RUNNING && state_ != AgentState::STARTING) {
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
        std::string cgroup_path = "/sys/fs/cgroup/agentos/" + config_.name + "_" + std::to_string(id_);

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

// ============================================================================
// AgentManager Implementation
// ============================================================================

AgentManager::AgentManager(const std::string& kernel_socket)
    : kernel_socket_(kernel_socket) {
    spdlog::debug("AgentManager initialized (socket={})", kernel_socket);
}

AgentManager::~AgentManager() {
    stop_all();
}

std::shared_ptr<AgentProcess> AgentManager::spawn_agent(const AgentConfig& config) {
    if (agents_by_name_.count(config.name)) {
        spdlog::error("Agent {} already exists", config.name);
        return nullptr;
    }

    // Set kernel socket if not specified
    AgentConfig final_config = config;
    if (final_config.socket_path.empty()) {
        final_config.socket_path = kernel_socket_;
    }

    auto agent = std::make_shared<AgentProcess>(final_config);

    if (!agent->start()) {
        return nullptr;
    }

    agents_by_name_[config.name] = agent;
    agents_by_id_[agent->id()] = agent;

    // Save config for potential restart if policy != NEVER
    if (config.restart.policy != RestartPolicy::NEVER) {
        saved_configs_[config.name] = final_config;

        // Initialize restart state if not exists
        if (restart_states_.find(config.name) == restart_states_.end()) {
            RestartState state;
            state.window_start = std::chrono::steady_clock::now();
            restart_states_[config.name] = state;
        }
    }

    return agent;
}

std::shared_ptr<AgentProcess> AgentManager::get_agent(const std::string& name) {
    auto it = agents_by_name_.find(name);
    return (it != agents_by_name_.end()) ? it->second : nullptr;
}

std::shared_ptr<AgentProcess> AgentManager::get_agent(uint32_t id) {
    auto it = agents_by_id_.find(id);
    return (it != agents_by_id_.end()) ? it->second : nullptr;
}

bool AgentManager::kill_agent(const std::string& name) {
    auto it = agents_by_name_.find(name);
    if (it == agents_by_name_.end()) {
        return false;
    }

    auto agent = it->second;
    agent->stop();

    agents_by_id_.erase(agent->id());
    agents_by_name_.erase(it);

    return true;
}

bool AgentManager::kill_agent(uint32_t id) {
    auto it = agents_by_id_.find(id);
    if (it == agents_by_id_.end()) {
        return false;
    }

    auto agent = it->second;
    agent->stop();

    agents_by_name_.erase(agent->name());
    agents_by_id_.erase(it);

    return true;
}

std::vector<std::shared_ptr<AgentProcess>> AgentManager::list_agents() const {
    std::vector<std::shared_ptr<AgentProcess>> result;
    for (const auto& [_, agent] : agents_by_name_) {
        result.push_back(agent);
    }
    return result;
}

void AgentManager::stop_all() {
    spdlog::info("Stopping all agents...");

    for (auto& [_, agent] : agents_by_name_) {
        agent->stop();
    }

    agents_by_name_.clear();
    agents_by_id_.clear();
}

void AgentManager::set_restart_event_callback(RestartEventCallback callback) {
    restart_event_callback_ = std::move(callback);
}

uint32_t AgentManager::calculate_backoff_delay(const RestartConfig& config, uint32_t consecutive_failures) {
    if (consecutive_failures == 0) {
        return config.backoff_initial_ms;
    }

    // Calculate exponential backoff: initial * multiplier^failures
    double delay = config.backoff_initial_ms;
    for (uint32_t i = 0; i < consecutive_failures; ++i) {
        delay *= config.backoff_multiplier;
        if (delay >= config.backoff_max_ms) {
            return config.backoff_max_ms;
        }
    }

    return static_cast<uint32_t>(delay);
}

void AgentManager::reap_and_restart_agents() {
    std::vector<std::string> dead_agents;

    for (auto& [name, agent] : agents_by_name_) {
        if (!agent->is_running() && agent->state() == AgentState::RUNNING) {
            // Agent died unexpectedly
            spdlog::warn("Agent {} died unexpectedly (exit_code={})", name, agent->exit_code());
            dead_agents.push_back(name);
        }
    }

    for (const auto& name : dead_agents) {
        auto agent = agents_by_name_[name];
        int exit_code = agent->exit_code();

        // Remove from active agents
        agents_by_id_.erase(agent->id());
        agents_by_name_.erase(name);

        // Check if we should restart
        auto config_it = saved_configs_.find(name);
        if (config_it == saved_configs_.end()) {
            // No restart policy configured
            spdlog::info("Agent {} exited, no restart policy", name);
            continue;
        }

        const AgentConfig& config = config_it->second;
        RestartState& state = restart_states_[name];

        // Check restart policy
        bool should_restart = false;
        switch (config.restart.policy) {
            case RestartPolicy::ALWAYS:
                should_restart = true;
                break;
            case RestartPolicy::ON_FAILURE:
                should_restart = (exit_code != 0);
                break;
            case RestartPolicy::NEVER:
            default:
                should_restart = false;
                break;
        }

        if (!should_restart) {
            spdlog::info("Agent {} exited with code {}, restart policy says no restart",
                name, exit_code);
            // Clean up saved config
            saved_configs_.erase(name);
            restart_states_.erase(name);
            continue;
        }

        // Check if we're within the restart window
        auto now = std::chrono::steady_clock::now();
        auto window_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - state.window_start).count();

        if (window_elapsed >= config.restart.restart_window_sec) {
            // Reset window
            state.window_start = now;
            state.restart_count = 0;
            state.consecutive_failures = 0;
            spdlog::debug("Agent {} restart window reset", name);
        }

        // Check if we've exceeded max restarts
        if (state.restart_count >= config.restart.max_restarts) {
            if (!state.escalated) {
                spdlog::error("Agent {} exceeded max_restarts ({}) within window, escalating",
                    name, config.restart.max_restarts);
                state.escalated = true;

                // Emit escalation event
                if (restart_event_callback_) {
                    restart_event_callback_("AGENT_ESCALATED", name, state.restart_count, exit_code);
                }
            }
            continue;
        }

        // Calculate backoff delay
        uint32_t backoff_ms = calculate_backoff_delay(config.restart, state.consecutive_failures);

        spdlog::info("Agent {} will restart in {}ms (attempt {}/{})",
            name, backoff_ms, state.restart_count + 1, config.restart.max_restarts);

        // Queue the restart
        PendingRestart pending;
        pending.agent_name = name;
        pending.scheduled_time = now + std::chrono::milliseconds(backoff_ms);
        pending.config = config;
        pending_restarts_.push_back(pending);

        // Update state
        state.restart_count++;
        state.consecutive_failures++;

        // Emit restarting event
        if (restart_event_callback_) {
            restart_event_callback_("AGENT_RESTARTING", name, state.restart_count, exit_code);
        }
    }
}

void AgentManager::process_pending_restarts() {
    if (pending_restarts_.empty()) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    std::vector<PendingRestart> still_pending;

    for (auto& pending : pending_restarts_) {
        if (now >= pending.scheduled_time) {
            // Time to restart this agent
            spdlog::info("Restarting agent: {} (scheduled restart)", pending.agent_name);

            auto agent = std::make_shared<AgentProcess>(pending.config);

            if (agent->start()) {
                agents_by_name_[pending.agent_name] = agent;
                agents_by_id_[agent->id()] = agent;

                spdlog::info("Agent {} restarted successfully (new id={}, pid={})",
                    pending.agent_name, agent->id(), agent->pid());

                // Reset consecutive failures on successful start
                auto state_it = restart_states_.find(pending.agent_name);
                if (state_it != restart_states_.end()) {
                    // Note: We don't reset consecutive_failures here because
                    // we want to track consecutive failures across restarts
                    // It gets reset when the window expires
                }
            } else {
                spdlog::error("Failed to restart agent {}", pending.agent_name);

                // The agent will be detected as dead again on next reap cycle
                // if the restart failed immediately
            }
        } else {
            // Not yet time, keep in pending
            still_pending.push_back(pending);
        }
    }

    pending_restarts_ = std::move(still_pending);
}

} // namespace agentos::runtime
