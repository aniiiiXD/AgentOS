#include "kernel/ipc_mailbox.hpp"

namespace clove::kernel {

RegisterResult AgentMailboxRegistry::register_name(uint32_t agent_id, const std::string& name) {
    RegisterResult result;
    if (name.empty()) {
        result.error = "name required";
        return result;
    }

    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = names_.find(name);
    if (it != names_.end() && it->second != agent_id) {
        result.error = "name already registered";
        return result;
    }

    names_[name] = agent_id;
    ids_to_names_[agent_id] = name;
    result.success = true;
    return result;
}

void AgentMailboxRegistry::unregister(uint32_t agent_id) {
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = ids_to_names_.find(agent_id);
        if (it != ids_to_names_.end()) {
            names_.erase(it->second);
            ids_to_names_.erase(it);
        }
    }
    {
        std::lock_guard<std::mutex> lock(mailbox_mutex_);
        mailboxes_.erase(agent_id);
    }
}

std::optional<uint32_t> AgentMailboxRegistry::resolve_name(const std::string& name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = names_.find(name);
    if (it == names_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::string AgentMailboxRegistry::get_name(uint32_t agent_id) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = ids_to_names_.find(agent_id);
    if (it == ids_to_names_.end()) {
        return {};
    }
    return it->second;
}

void AgentMailboxRegistry::enqueue(uint32_t target_id, const IPCMessage& message) {
    std::lock_guard<std::mutex> lock(mailbox_mutex_);
    mailboxes_[target_id].push(message);
}

std::vector<IPCMessage> AgentMailboxRegistry::dequeue(uint32_t agent_id, int max_messages) {
    std::vector<IPCMessage> messages;
    if (max_messages <= 0) {
        return messages;
    }

    std::lock_guard<std::mutex> lock(mailbox_mutex_);
    auto& mailbox = mailboxes_[agent_id];
    int count = 0;

    while (!mailbox.empty() && count < max_messages) {
        messages.push_back(mailbox.front());
        mailbox.pop();
        count++;
    }

    return messages;
}

int AgentMailboxRegistry::broadcast(const IPCMessage& message, bool include_self) {
    std::lock_guard<std::mutex> reg_lock(registry_mutex_);
    std::lock_guard<std::mutex> mail_lock(mailbox_mutex_);

    int delivered_count = 0;
    for (const auto& [agent_id, name] : ids_to_names_) {
        if (agent_id == message.from_id && !include_self) {
            continue;
        }

        mailboxes_[agent_id].push(message);
        delivered_count++;
    }

    return delivered_count;
}

} // namespace clove::kernel
