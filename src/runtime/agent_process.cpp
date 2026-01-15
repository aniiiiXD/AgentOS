#include "runtime/agent_process.hpp"
#include <spdlog/spdlog.h>
#include <atomic>

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

void AgentManager::reap_agents() {
    std::vector<std::string> dead_agents;

    for (auto& [name, agent] : agents_by_name_) {
        if (!agent->is_running() && agent->state() == AgentState::RUNNING) {
            // Agent died unexpectedly
            spdlog::warn("Agent {} died unexpectedly", name);
            dead_agents.push_back(name);
        }
    }

    for (const auto& name : dead_agents) {
        auto agent = agents_by_name_[name];
        agents_by_id_.erase(agent->id());
        agents_by_name_.erase(name);
    }
}

} // namespace agentos::runtime
