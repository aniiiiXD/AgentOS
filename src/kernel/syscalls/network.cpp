#include "kernel/syscall_handlers.hpp"
#include "kernel/syscall_router.hpp"
#include "kernel/async_task_manager.hpp"
#include "kernel/async_helpers.hpp"
#include "kernel/permissions_store.hpp"
#include "worlds/world_engine.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <thread>

using json = nlohmann::json;

namespace clove::kernel {

void NetworkSyscalls::register_syscalls(SyscallRouter& router) {
    router.register_handler(ipc::SyscallOp::SYS_HTTP,
        [this](const ipc::Message& msg) { return handle_http(msg); });
}

ipc::Message NetworkSyscalls::http_sync(KernelContext& context, const ipc::Message& msg, const json& j) {
    auto& perms = context.permissions_store.get_or_create(msg.agent_id);
    std::string method = j.value("method", "GET");
    std::string url = j.value("url", "");
    int timeout = j.value("timeout", 30);

    if (url.empty()) {
        json response;
        response["success"] = false;
        response["error"] = "URL required";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
    }

    if (!perms.can_http) {
        spdlog::warn("Agent {} denied HTTP access (permission denied)", msg.agent_id);
        json response;
        response["success"] = false;
        response["error"] = "Permission denied: HTTP not allowed";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
    }

    if (!perms.can_http_method(method)) {
        json response;
        response["success"] = false;
        response["error"] = "Permission denied: HTTP method not allowed";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
    }

    std::string domain = PermissionChecker::extract_domain(url);
    if (!perms.can_access_domain(domain)) {
        spdlog::warn("Agent {} denied access to domain: {}", msg.agent_id, domain);
        json response;
        response["success"] = false;
        response["error"] = "Permission denied: domain not in whitelist: " + domain;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
    }

    spdlog::debug("Agent {} making {} request to {}", msg.agent_id, method, url);

    std::string curl_cmd = "curl -s -X " + method;
    curl_cmd += " --max-time " + std::to_string(timeout);

    if (j.contains("headers") && j["headers"].is_object()) {
        for (auto& [key, val] : j["headers"].items()) {
            curl_cmd += " -H '" + key + ": " + val.get<std::string>() + "'";
        }
    }

    if (j.contains("body") && (method == "POST" || method == "PUT" || method == "PATCH")) {
        std::string body = j["body"].get<std::string>();
        curl_cmd += " -d '" + body + "'";
    }

    curl_cmd += " '" + url + "' 2>&1";

    FILE* pipe = popen(curl_cmd.c_str(), "r");
    if (!pipe) {
        json response;
        response["success"] = false;
        response["error"] = "Failed to execute HTTP request";
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
    }

    std::string output;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

    int status = pclose(pipe);
    bool success = (WIFEXITED(status) && WEXITSTATUS(status) == 0);

    json response;
    response["success"] = success;
    response["body"] = output;
    response["status_code"] = success ? 200 : 0;

    return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
}

ipc::Message NetworkSyscalls::handle_http(const ipc::Message& msg) {
    if (context_.world_engine.is_agent_in_world(msg.agent_id)) {
        auto world_id = context_.world_engine.get_agent_world(msg.agent_id);
        if (world_id) {
            auto* world = context_.world_engine.get_world(*world_id);
            if (world && world->network().is_enabled()) {
                try {
                    json j = json::parse(msg.payload_str());
                    std::string url = j.value("url", "");
                    if (world->network().should_intercept(url)) {
                        return handle_http_virtual(msg, world);
                    }
                } catch (...) {
                }
            }
        }
    }

    json j;
    try {
        j = json::parse(msg.payload_str());
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse HTTP request: {}", e.what());
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
    }

    bool async = async_helpers::should_async(j, true);
    if (async) {
        std::string url = j.value("url", "");
        if (url.empty()) {
            json response;
            response["success"] = false;
            response["error"] = "URL required";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }

        auto& perms = context_.permissions_store.get_or_create(msg.agent_id);
        if (!perms.can_http) {
            json response;
            response["success"] = false;
            response["error"] = "Permission denied: HTTP not allowed";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }
        std::string method = j.value("method", "GET");
        if (!perms.can_http_method(method)) {
            json response;
            response["success"] = false;
            response["error"] = "Permission denied: HTTP method not allowed";
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }

        std::string domain = PermissionChecker::extract_domain(url);
        if (!perms.can_access_domain(domain)) {
            json response;
            response["success"] = false;
            response["error"] = "Permission denied: domain not in whitelist: " + domain;
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }

        return async_helpers::submit_async(context_, msg, j, http_sync);
    }

    return http_sync(context_, msg, j);
}

ipc::Message NetworkSyscalls::handle_http_virtual(const ipc::Message& msg, clove::worlds::World* world) {
    try {
        json j = json::parse(msg.payload_str());
        std::string method = j.value("method", "GET");
        std::string url = j.value("url", "");

        world->record_syscall();

        if (world->chaos().should_fail_network(url)) {
            spdlog::debug("Chaos: Injected network failure for {} in world '{}'", url, world->id());
            json response;
            response["success"] = false;
            response["error"] = "Simulated network failure (chaos)";
            response["body"] = "";
            response["status_code"] = 503;
            response["world"] = world->id();
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }

        auto mock_response = world->network().get_response(url, method);

        if (!mock_response) {
            json response;
            response["success"] = false;
            response["error"] = "No mock response configured for URL";
            response["body"] = "";
            response["status_code"] = 0;
            response["world"] = world->id();
            return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
        }

        uint32_t total_latency = mock_response->latency_ms + world->chaos().get_latency();
        if (total_latency > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(total_latency));
        }

        spdlog::debug("Agent {} got mock HTTP response for {} in world '{}': status={}",
                      msg.agent_id, url, world->id(), mock_response->status_code);

        json response;
        response["success"] = (mock_response->status_code >= 200 && mock_response->status_code < 400);
        response["body"] = mock_response->body;
        response["status_code"] = mock_response->status_code;
        response["headers"] = mock_response->headers;
        response["world"] = world->id();
        response["mocked"] = true;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());

    } catch (const std::exception& e) {
        json response;
        response["success"] = false;
        response["error"] = std::string("invalid request: ") + e.what();
        response["body"] = "";
        response["status_code"] = 0;
        return ipc::Message(msg.agent_id, ipc::SyscallOp::SYS_HTTP, response.dump());
    }
}

} // namespace clove::kernel
