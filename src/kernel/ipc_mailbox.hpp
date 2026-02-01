#pragma once
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <mutex>
#include <string>
#include <vector>
#include <optional>
#include <chrono>
#include <nlohmann/json.hpp>

namespace clove::kernel {

// IPC Message for agent-to-agent communication
struct IPCMessage {
    uint32_t from_id;
    std::string from_name;
    nlohmann::json message;
    std::chrono::steady_clock::time_point timestamp;
};

struct RegisterResult {
    bool success = false;
    std::string error;
};

class AgentMailboxRegistry {
public:
    RegisterResult register_name(uint32_t agent_id, const std::string& name);
    void unregister(uint32_t agent_id);
    std::optional<uint32_t> resolve_name(const std::string& name) const;
    std::string get_name(uint32_t agent_id) const;

    void enqueue(uint32_t target_id, const IPCMessage& message);
    std::vector<IPCMessage> dequeue(uint32_t agent_id, int max_messages);
    int broadcast(const IPCMessage& message, bool include_self);

private:
    mutable std::mutex registry_mutex_;
    std::unordered_map<std::string, uint32_t> names_;
    std::unordered_map<uint32_t, std::string> ids_to_names_;

    mutable std::mutex mailbox_mutex_;
    std::unordered_map<uint32_t, std::queue<IPCMessage>> mailboxes_;
};

} // namespace clove::kernel
