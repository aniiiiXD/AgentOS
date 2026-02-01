#include "kernel/syscall_handlers.hpp"
#include "kernel/syscall_router.hpp"
#include "kernel/audit_log.hpp"
#include "runtime/agent/manager.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace clove::kernel {

void AgentSyscalls::register_syscalls(SyscallRouter& router) {
    router.register_handler(ipc::SyscallOp::SYS_SPAWN, [this](const ipc::Message& msg) { return handle_spawn(msg); });
    router.register_handler(ipc::SyscallOp::SYS_KILL, [this](const ipc::Message& msg) { return handle_kill(msg); });
    router.register_handler(ipc::SyscallOp::SYS_LIST, [this](const ipc::Message& msg) { return handle_list(msg); });
    router.register_handler(ipc::SyscallOp::SYS_PAUSE, [this](const ipc::Message& msg) { return handle_pause(msg); });
    router.register_handler(ipc::SyscallOp::SYS_RESUME, [this](const ipc::Message& msg) { return handle_resume(msg); });
}

ipc::Message AgentSyscalls::handle_spawn(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        runtime::AgentConfig config;
        // Use ternary to avoid eager evaluation of generate_id() when name is provided
        config.name = j.contains("name") ? j["name"].get<std::string>()
            : "agent_" + std::to_string(runtime::AgentProcess::generate_id());
        config.script_path = j.value("script", "");
        config.python_path = j.value("python", "python3");
        config.sandboxed = context_.config.enable_sandboxing && j.value("sandboxed", true);
        config.enable_network = j.value("network", false);

        if (j.contains("limits")) {
            auto& lim = j["limits"];
            config.limits.memory_limit_bytes = lim.value("memory", 256 * 1024 * 1024);
            config.limits.max_pids = lim.value("max_pids", 64);
            config.limits.cpu_quota_us = lim.value("cpu_quota", 100000);
            config.limits.cpu_period_us = lim.value("cpu_period", 100000);
            config.limits.cpu_shares = lim.value("cpu_shares", 1024);
        }

        config.restart.policy = runtime::restart_policy_from_string(
            j.value("restart_policy", "never"));
        config.restart.max_restarts = j.value("max_restarts", 5);
        config.restart.restart_window_sec = j.value("restart_window", 300);

        if (config.script_path.empty()) {
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SPAWN,
                R"({"error": "script path required"})");
        }

        spdlog::info("Spawning agent: {} (script={})", config.name, config.script_path);

        auto agent = context_.agent_manager.spawn_agent(config);
        if (!agent) {
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SPAWN,
                R"({"error": "failed to spawn agent"})");
        }

        agent->set_parent_id(msg.agent_id);

        if (msg.agent_id > 0) {
            auto parent = context_.agent_manager.get_agent(msg.agent_id);
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

        auto iso_status = agent->get_isolation_status();
        json isolation;
        isolation["fully_isolated"] = iso_status.fully_isolated;
        isolation["namespaces"]["pid"] = iso_status.pid_namespace;
        isolation["namespaces"]["net"] = iso_status.net_namespace;
        isolation["namespaces"]["mnt"] = iso_status.mnt_namespace;
        isolation["namespaces"]["uts"] = iso_status.uts_namespace;
        isolation["cgroups"]["available"] = iso_status.cgroups_available;
        isolation["cgroups"]["memory_limit"] = iso_status.memory_limit_applied;
        isolation["cgroups"]["cpu_quota"] = iso_status.cpu_quota_applied;
        isolation["cgroups"]["pids_limit"] = iso_status.pids_limit_applied;
        if (!iso_status.degraded_reason.empty()) {
            isolation["degraded_reason"] = iso_status.degraded_reason;
        }
        response["isolation"] = isolation;

        json event_data;
        event_data["agent_id"] = agent->id();
        event_data["name"] = agent->name();
        event_data["pid"] = agent->pid();
        event_data["parent_id"] = msg.agent_id;
        context_.event_bus.emit(KernelEventType::AGENT_SPAWNED, event_data, 0);

        context_.audit_logger.log_lifecycle("AGENT_SPAWNED", agent->id(), agent->name(), event_data);

        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SPAWN, response.dump());

    } catch (const std::exception& e) {
        spdlog::error("Failed to parse spawn request: {}", e.what());
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_SPAWN,
            R"({"error": "invalid JSON"})");
    }
}

ipc::Message AgentSyscalls::handle_kill(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        bool killed = false;
        uint32_t target_id = 0;
        std::string target_name;

        if (j.contains("id")) {
            target_id = j["id"].get<uint32_t>();
            auto agent = context_.agent_manager.get_agent(target_id);
            if (agent) target_name = agent->name();
            killed = context_.agent_manager.kill_agent(target_id);
        } else if (j.contains("name")) {
            target_name = j["name"].get<std::string>();
            auto agent = context_.agent_manager.get_agent(target_name);
            if (agent) target_id = agent->id();
            killed = context_.agent_manager.kill_agent(target_name);
        }

        if (killed && target_id > 0) {
            json event_data;
            event_data["agent_id"] = target_id;
            event_data["name"] = target_name;
            event_data["killed_by"] = msg.agent_id;
            context_.event_bus.emit(KernelEventType::AGENT_EXITED, event_data, 0);

            context_.audit_logger.log_lifecycle("AGENT_KILLED", target_id, target_name, event_data);
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

ipc::Message AgentSyscalls::handle_list(const ipc::Message& msg) {
    json response = json::array();

    for (const auto& agent : context_.agent_manager.list_agents()) {
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

ipc::Message AgentSyscalls::handle_pause(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        bool paused = false;
        uint32_t target_id = 0;
        std::string target_name;

        if (j.contains("id")) {
            target_id = j["id"].get<uint32_t>();
            auto agent = context_.agent_manager.get_agent(target_id);
            if (agent) target_name = agent->name();
            paused = context_.agent_manager.pause_agent(target_id);
        } else if (j.contains("name")) {
            target_name = j["name"].get<std::string>();
            auto agent = context_.agent_manager.get_agent(target_name);
            if (agent) target_id = agent->id();
            paused = context_.agent_manager.pause_agent(target_name);
        }

        if (paused && target_id > 0) {
            json event_data;
            event_data["agent_id"] = target_id;
            event_data["name"] = target_name;
            event_data["paused_by"] = msg.agent_id;
            context_.event_bus.emit(KernelEventType::AGENT_PAUSED, event_data, 0);

            context_.audit_logger.log_lifecycle("AGENT_PAUSED", target_id, target_name, event_data);
        }

        json response;
        response["success"] = paused;
        response["paused"] = paused;
        response["agent_id"] = target_id;

        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_PAUSE, response.dump());

    } catch (const std::exception& e) {
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_PAUSE,
            R"({"success": false, "error": "invalid request"})");
    }
}

ipc::Message AgentSyscalls::handle_resume(const ipc::Message& msg) {
    try {
        json j = json::parse(msg.payload_str());

        bool resumed = false;
        uint32_t target_id = 0;
        std::string target_name;

        if (j.contains("id")) {
            target_id = j["id"].get<uint32_t>();
            auto agent = context_.agent_manager.get_agent(target_id);
            if (agent) target_name = agent->name();
            resumed = context_.agent_manager.resume_agent(target_id);
        } else if (j.contains("name")) {
            target_name = j["name"].get<std::string>();
            auto agent = context_.agent_manager.get_agent(target_name);
            if (agent) target_id = agent->id();
            resumed = context_.agent_manager.resume_agent(target_name);
        }

        if (resumed && target_id > 0) {
            json event_data;
            event_data["agent_id"] = target_id;
            event_data["name"] = target_name;
            event_data["resumed_by"] = msg.agent_id;
            context_.event_bus.emit(KernelEventType::AGENT_RESUMED, event_data, 0);

            context_.audit_logger.log_lifecycle("AGENT_RESUMED", target_id, target_name, event_data);
        }

        json response;
        response["success"] = resumed;
        response["resumed"] = resumed;
        response["agent_id"] = target_id;

        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_RESUME, response.dump());

    } catch (const std::exception& e) {
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_RESUME,
            R"({"success": false, "error": "invalid request"})");
    }
}

} // namespace clove::kernel
