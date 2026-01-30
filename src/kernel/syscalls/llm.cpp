#include "kernel/syscall_handlers.hpp"
#include "kernel/syscall_router.hpp"
#include "kernel/permissions_store.hpp"
#include "kernel/async_helpers.hpp"
#include "kernel/llm_queue.hpp"
#include "runtime/agent/manager.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace clove::kernel {

void LlmSyscalls::register_syscalls(SyscallRouter& router) {
    router.register_handler(ipc::SyscallOp::SYS_THINK,
        [this](const ipc::Message& msg) { return handle_think(msg); });
    router.register_handler(ipc::SyscallOp::SYS_LLM_REPORT,
        [this](const ipc::Message& msg) { return handle_report(msg); });
}

ipc::Message LlmSyscalls::think_sync(KernelContext& context, const ipc::Message& msg) {
    auto& perms = context.permissions_store.get_or_create(msg.agent_id);

    if (!perms.can_use_llm()) {
        spdlog::warn("Agent {} denied LLM access (quota exceeded or permission denied)", msg.agent_id);
        json response;
        response["success"] = false;
        response["error"] = "Permission denied: LLM quota exceeded or not allowed";
        response["content"] = "";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_THINK, response.dump());
    }

    std::string payload = msg.payload_str();
    spdlog::debug("Agent {} thinking: {}", msg.agent_id,
        payload.length() > 50 ? payload.substr(0, 50) + "..." : payload);

    auto future = context.llm_queue.submit(msg.agent_id, payload);
    auto result = future.get();

    if (result.success && result.tokens_used > 0) {
        auto agent = context.agent_manager.get_agent(msg.agent_id);
        if (agent) {
            agent->record_llm_call(result.tokens_used);
        }
        perms.record_llm_usage(result.tokens_used);
    }

    json response;
    response["success"] = result.success;
    if (result.success) {
        response["content"] = result.content;
        response["tokens"] = result.tokens_used;
        spdlog::debug("LLM response: {} tokens", result.tokens_used);
    } else {
        response["error"] = result.error;
        response["content"] = "";
        spdlog::warn("LLM error for agent {}: {}", msg.agent_id, result.error);
    }

    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_THINK, response.dump());
}

ipc::Message LlmSyscalls::handle_think(const ipc::Message& msg) {
    json j;
    try {
        if (!msg.payload.empty()) {
            j = json::parse(msg.payload_str());
        } else {
            j = json::object();
        }
    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        response["content"] = "";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_THINK, response.dump());
    }

    bool async = async_helpers::should_async(j, true);
    if (async) {
        return async_helpers::submit_async(context_, msg, j,
            [](KernelContext& context, const ipc::Message& task_msg, const json&) {
                return think_sync(context, task_msg);
            });
    }

    return think_sync(context_, msg);
}

ipc::Message LlmSyscalls::handle_report(const ipc::Message& msg) {
    json j;
    try {
        j = json::parse(msg.payload_str());
    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_LLM_REPORT, response.dump());
    }

    uint32_t tokens = 0;
    if (j.contains("tokens")) {
        tokens = j["tokens"].get<uint32_t>();
    }

    auto& perms = context_.permissions_store.get_or_create(msg.agent_id);
    bool allowed = perms.can_use_llm(tokens);

    if (tokens > 0) {
        perms.record_llm_usage(tokens);
        auto agent = context_.agent_manager.get_agent(msg.agent_id);
        if (agent) {
            agent->record_llm_call(tokens);
        }
    }

    json response;
    response["success"] = allowed;
    response["tokens"] = tokens;
    response["quota_exceeded"] = !allowed;
    if (!allowed) {
        response["error"] = "LLM quota exceeded or permission denied";
    }
    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_LLM_REPORT, response.dump());
}

} // namespace clove::kernel
