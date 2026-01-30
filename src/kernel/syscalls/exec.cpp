#include "kernel/syscall_handlers.hpp"
#include "kernel/syscall_router.hpp"
#include "kernel/async_task_manager.hpp"
#include "kernel/async_helpers.hpp"
#include "kernel/permissions_store.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <cstdio>
#include <sys/wait.h>

using json = nlohmann::json;

namespace clove::kernel {

void ExecSyscalls::register_syscalls(SyscallRouter& router) {
    router.register_handler(ipc::SyscallOp::SYS_EXEC,
        [this](const ipc::Message& msg) { return handle_exec(msg); });
}

ipc::Message ExecSyscalls::exec_sync(KernelContext& context, const ipc::Message& msg, const json& j) {
    auto& perms = context.permissions_store.get_or_create(msg.agent_id);
    std::string command = j.value("command", "");
    std::string cwd = j.value("cwd", "");
    int timeout_sec = j.value("timeout", 30);
    (void)timeout_sec;

    if (command.empty()) {
        json response;
        response["success"] = false;
        response["error"] = "command required";
        response["stdout"] = "";
        response["stderr"] = "";
        response["exit_code"] = -1;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EXEC, response.dump());
    }

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

    std::string full_command = command;
    if (!cwd.empty()) {
        full_command = "cd " + cwd + " && " + command;
    }
    full_command += " 2>&1";

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

    std::string output;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

    int status = pclose(pipe);
    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

    json response;
    response["success"] = (exit_code == 0);
    response["stdout"] = output;
    response["stderr"] = "";
    response["exit_code"] = exit_code;

    spdlog::debug("Command exit code: {}", exit_code);
    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EXEC, response.dump());
}

ipc::Message ExecSyscalls::handle_exec(const ipc::Message& msg) {
    json j;
    try {
        j = json::parse(msg.payload_str());
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

    bool async = async_helpers::should_async(j, true);
    if (async) {
        std::string command = j.value("command", "");
        if (command.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "command required";
            response["stdout"] = "";
            response["stderr"] = "";
            response["exit_code"] = -1;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EXEC, response.dump());
        }

        auto& perms = context_.permissions_store.get_or_create(msg.agent_id);
        if (!perms.can_execute_command(command)) {
            json response;
            response["success"] = false;
            response["error"] = "Permission denied: command not allowed";
            response["stdout"] = "";
            response["stderr"] = "";
            response["exit_code"] = -1;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_EXEC, response.dump());
        }

        return async_helpers::submit_async(context_, msg, j, exec_sync);
    }

    return exec_sync(context_, msg, j);
}

} // namespace clove::kernel
