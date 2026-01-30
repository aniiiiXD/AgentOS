#pragma once
#include <functional>
#include <string>
#include <nlohmann/json.hpp>
#include "ipc/protocol.hpp"
#include "kernel/context.hpp"
#include "kernel/async_task_manager.hpp"

namespace clove::kernel::async_helpers {

using json = nlohmann::json;
using SyncFn = std::function<ipc::Message(KernelContext&, const ipc::Message&, const json&)>;

inline bool should_async(const json& j, bool default_async) {
    if (j.contains("async")) {
        return j.value("async", default_async);
    }
    return default_async;
}

inline ipc::Message submit_async(KernelContext& context, const ipc::Message& msg,
                                 const json& request, SyncFn sync_fn) {
    uint64_t request_id = request.value("request_id", 0ULL);
    if (request_id == 0) {
        request_id = context.async_tasks.next_request_id();
    }

    std::string payload = msg.payload_str();
    context.async_tasks.submit(msg.agent_id, msg.opcode, request_id,
        [payload, agent_id = msg.agent_id, opcode = msg.opcode, &context, sync_fn]() {
            try {
                json task_json = json::parse(payload);
                task_json["async"] = false;
                ipc::Message task_msg(agent_id, opcode, task_json.dump());
                return sync_fn(context, task_msg, task_json);
            } catch (const std::exception& e) {
                json response;
                response["success"] = false;
                response["error"] = std::string("invalid request: ") + e.what();
                return ipc::Message(agent_id, opcode, response.dump());
            }
        });

    json response;
    response["success"] = true;
    response["async"] = true;
    response["request_id"] = request_id;
    response["status"] = "accepted";
    return ipc::Message(msg.agent_id, msg.opcode, response.dump());
}

} // namespace clove::kernel::async_helpers
