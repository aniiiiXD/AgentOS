#include "kernel/permissions.hpp"
#include <spdlog/spdlog.h>
#include <fnmatch.h>
#include <algorithm>
#include <regex>
#include <filesystem>
#include <strings.h>

namespace fs = std::filesystem;

namespace clove::kernel {

// ============================================================================
// AgentPermissions Implementation
// ============================================================================

AgentPermissions AgentPermissions::from_json(const nlohmann::json& j) {
    AgentPermissions perms;

    // Syscall permissions
    if (j.contains("can_exec")) perms.can_exec = j["can_exec"].get<bool>();
    if (j.contains("can_read")) perms.can_read = j["can_read"].get<bool>();
    if (j.contains("can_write")) perms.can_write = j["can_write"].get<bool>();
    if (j.contains("can_think")) perms.can_think = j["can_think"].get<bool>();
    if (j.contains("can_spawn")) perms.can_spawn = j["can_spawn"].get<bool>();
    if (j.contains("can_http")) perms.can_http = j["can_http"].get<bool>();

    // Filesystem restrictions
    if (j.contains("filesystem")) {
        auto& fs = j["filesystem"];
        if (fs.contains("read")) {
            for (const auto& p : fs["read"]) {
                perms.allowed_read_paths.push_back(p.get<std::string>());
            }
        }
        if (fs.contains("write")) {
            for (const auto& p : fs["write"]) {
                perms.allowed_write_paths.push_back(p.get<std::string>());
            }
        }
        if (fs.contains("blocked")) {
            for (const auto& p : fs["blocked"]) {
                perms.blocked_paths.push_back(p.get<std::string>());
            }
        }
    }

    // Command restrictions
    if (j.contains("exec")) {
        for (const auto& cmd : j["exec"]) {
            perms.allowed_commands.push_back(cmd.get<std::string>());
        }
        perms.can_exec = true;
    }
    if (j.contains("blocked_commands")) {
        for (const auto& cmd : j["blocked_commands"]) {
            perms.blocked_commands.push_back(cmd.get<std::string>());
        }
    }

    // Network restrictions
    if (j.contains("network")) {
        for (const auto& domain : j["network"]) {
            perms.allowed_domains.push_back(domain.get<std::string>());
        }
        if (!perms.allowed_domains.empty()) {
            perms.can_http = true;
        }
    }
    if (j.contains("http_methods")) {
        for (const auto& method : j["http_methods"]) {
            perms.allowed_http_methods.push_back(method.get<std::string>());
        }
        if (!perms.allowed_http_methods.empty()) {
            perms.can_http = true;
        }
    }

    // Resource limits
    if (j.contains("llm")) {
        auto& llm = j["llm"];
        if (llm.contains("max_tokens")) perms.max_llm_tokens = llm["max_tokens"].get<uint64_t>();
        if (llm.contains("max_calls")) perms.max_llm_calls = llm["max_calls"].get<uint32_t>();
    }

    if (j.contains("max_exec_time_ms")) {
        perms.max_exec_time_ms = j["max_exec_time_ms"].get<uint64_t>();
    }

    return perms;
}

nlohmann::json AgentPermissions::to_json() const {
    nlohmann::json j;

    j["can_exec"] = can_exec;
    j["can_read"] = can_read;
    j["can_write"] = can_write;
    j["can_think"] = can_think;
    j["can_spawn"] = can_spawn;
    j["can_http"] = can_http;

    j["filesystem"]["read"] = allowed_read_paths;
    j["filesystem"]["write"] = allowed_write_paths;
    j["filesystem"]["blocked"] = blocked_paths;

    j["exec"] = allowed_commands;
    j["blocked_commands"] = blocked_commands;
    j["network"] = allowed_domains;
    j["http_methods"] = allowed_http_methods;

    j["llm"]["max_tokens"] = max_llm_tokens;
    j["llm"]["max_calls"] = max_llm_calls;
    j["llm"]["tokens_used"] = llm_tokens_used;
    j["llm"]["calls_made"] = llm_calls_made;

    j["max_exec_time_ms"] = max_exec_time_ms;

    return j;
}

AgentPermissions AgentPermissions::from_level(PermissionLevel level) {
    AgentPermissions perms;

    // Add default blocked paths and commands for all levels
    perms.blocked_paths = DEFAULT_BLOCKED_PATHS;
    perms.blocked_commands = DEFAULT_BLOCKED_COMMANDS;

    switch (level) {
        case PermissionLevel::UNRESTRICTED:
            perms.can_exec = true;
            perms.can_read = true;
            perms.can_write = true;
            perms.can_think = true;
            perms.can_spawn = true;
            perms.can_http = true;
            perms.blocked_paths.clear();  // Clear blocks for unrestricted
            perms.blocked_commands.clear();
            perms.allowed_domains.push_back("*");  // Allow all domains for unrestricted
            perms.allowed_http_methods.push_back("*");  // Allow all methods for unrestricted
            perms.allowed_commands.push_back("*");  // Allow all commands for unrestricted
            break;

        case PermissionLevel::STANDARD:
            perms.can_exec = false;
            perms.can_read = true;
            perms.can_write = true;
            perms.can_think = true;
            perms.can_spawn = false;
            perms.can_http = false;
            break;

        case PermissionLevel::SANDBOXED:
            perms.can_exec = false;
            perms.can_read = true;
            perms.can_write = true;
            perms.can_think = true;
            perms.can_spawn = false;
            perms.can_http = false;
            // Additional restrictions for sandboxed
            perms.allowed_read_paths = {"/tmp/*", "/home/*"};
            perms.allowed_write_paths = {"/tmp/*"};
            break;

        case PermissionLevel::READONLY:
            perms.can_exec = false;
            perms.can_read = true;
            perms.can_write = false;
            perms.can_think = true;
            perms.can_spawn = false;
            perms.can_http = false;
            break;

        case PermissionLevel::MINIMAL:
            perms.can_exec = false;
            perms.can_read = false;
            perms.can_write = false;
            perms.can_think = true;
            perms.can_spawn = false;
            perms.can_http = false;
            break;
    }

    return perms;
}

bool AgentPermissions::can_read_path(const std::string& path) const {
    if (!can_read) return false;

    // Normalize path
    std::string normalized = path;
    try {
        if (fs::exists(path)) {
            normalized = fs::canonical(path).string();
        }
    } catch (...) {
        // Use original path if canonicalization fails
    }

    // Check blocked paths first
    for (const auto& blocked : blocked_paths) {
        if (PermissionChecker::path_matches(normalized, blocked)) {
            spdlog::debug("Path blocked: {} matches {}", path, blocked);
            return false;
        }
    }

    // If no allowed paths specified, allow all (except blocked)
    if (allowed_read_paths.empty()) {
        return true;
    }

    // Check if path matches any allowed pattern
    for (const auto& allowed : allowed_read_paths) {
        if (PermissionChecker::path_matches(normalized, allowed)) {
            return true;
        }
    }

    return false;
}

bool AgentPermissions::can_write_path(const std::string& path) const {
    if (!can_write) return false;

    // Normalize path
    std::string normalized = path;
    try {
        auto parent = fs::path(path).parent_path();
        if (fs::exists(parent)) {
            normalized = (fs::canonical(parent) / fs::path(path).filename()).string();
        }
    } catch (...) {
        // Use original path if canonicalization fails
    }

    // Check blocked paths first
    for (const auto& blocked : blocked_paths) {
        if (PermissionChecker::path_matches(normalized, blocked)) {
            spdlog::debug("Path blocked for write: {} matches {}", path, blocked);
            return false;
        }
    }

    // If no allowed paths specified, allow all (except blocked)
    if (allowed_write_paths.empty()) {
        return true;
    }

    // Check if path matches any allowed pattern
    for (const auto& allowed : allowed_write_paths) {
        if (PermissionChecker::path_matches(normalized, allowed)) {
            return true;
        }
    }

    return false;
}

bool AgentPermissions::can_execute_command(const std::string& command) const {
    if (!can_exec) return false;

    // Check blocked commands first
    for (const auto& blocked : blocked_commands) {
        if (PermissionChecker::command_matches(command, blocked)) {
            spdlog::warn("Command blocked: {} matches {}", command, blocked);
            return false;
        }
    }

    // If no allowed commands specified, deny all
    if (allowed_commands.empty()) {
        return false;
    }

    // Check if command starts with any allowed prefix
    for (const auto& allowed : allowed_commands) {
        if (allowed == "*") {
            return true;
        }
        if (PermissionChecker::command_matches(command, allowed)) {
            return true;
        }
    }

    return false;
}

bool AgentPermissions::can_access_domain(const std::string& domain) const {
    if (!can_http) return false;

    // If no allowed domains specified, deny all
    if (allowed_domains.empty()) {
        return false;
    }

    // Check if domain matches any allowed pattern
    for (const auto& allowed : allowed_domains) {
        if (PermissionChecker::domain_matches(domain, allowed)) {
            return true;
        }
    }

    return false;
}

bool AgentPermissions::can_http_method(const std::string& method) const {
    if (!can_http) return false;
    if (allowed_http_methods.empty()) {
        return false;
    }
    for (const auto& allowed : allowed_http_methods) {
        if (allowed == "*") {
            return true;
        }
        if (strcasecmp(allowed.c_str(), method.c_str()) == 0) {
            return true;
        }
    }
    return false;
}

bool AgentPermissions::can_use_llm(uint32_t estimated_tokens) const {
    if (!can_think) return false;

    // Check call limit
    if (max_llm_calls > 0 && llm_calls_made >= max_llm_calls) {
        spdlog::debug("LLM call limit reached: {}/{}", llm_calls_made, max_llm_calls);
        return false;
    }

    // Check token limit
    if (max_llm_tokens > 0 && (llm_tokens_used + estimated_tokens) > max_llm_tokens) {
        spdlog::debug("LLM token limit would be exceeded: {} + {} > {}",
            llm_tokens_used, estimated_tokens, max_llm_tokens);
        return false;
    }

    return true;
}

void AgentPermissions::record_llm_usage(uint32_t tokens) {
    llm_calls_made++;
    llm_tokens_used += tokens;
}

// ============================================================================
// PermissionChecker Implementation
// ============================================================================

bool PermissionChecker::path_matches(const std::string& path, const std::string& pattern) {
    // Expand ~ to home directory
    std::string expanded_pattern = pattern;
    if (!pattern.empty() && pattern[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) {
            expanded_pattern = std::string(home) + pattern.substr(1);
        }
    }

    auto has_parent_ref = [](const std::string& value) {
        for (const auto& part : fs::path(value)) {
            if (part == "..") {
                return true;
            }
        }
        return false;
    };

    if (has_parent_ref(path) || has_parent_ref(expanded_pattern)) {
        return false;
    }

    // Use fnmatch for glob-style matching
    return fnmatch(expanded_pattern.c_str(), path.c_str(), FNM_PATHNAME | FNM_NOESCAPE) == 0;
}

bool PermissionChecker::command_matches(const std::string& command, const std::string& prefix) {
    // Check if command contains the blocked pattern anywhere
    // This catches things like "sudo rm -rf /" when blocking "sudo"
    if (command.find(prefix) != std::string::npos) {
        return true;
    }

    // Also check if command starts with prefix (for allowed commands)
    if (command.rfind(prefix, 0) == 0) {
        return true;
    }

    return false;
}

std::string PermissionChecker::extract_domain(const std::string& url) {
    // Simple domain extraction from URL
    std::string domain = url;

    // Remove protocol
    size_t proto_end = url.find("://");
    if (proto_end != std::string::npos) {
        domain = url.substr(proto_end + 3);
    }

    // Remove path
    size_t path_start = domain.find('/');
    if (path_start != std::string::npos) {
        domain = domain.substr(0, path_start);
    }

    // Remove port
    size_t port_start = domain.find(':');
    if (port_start != std::string::npos) {
        domain = domain.substr(0, port_start);
    }

    return domain;
}

bool PermissionChecker::domain_matches(const std::string& domain, const std::string& pattern) {
    // Universal wildcard - match all domains
    if (pattern == "*") {
        return true;
    }

    // Exact match
    if (domain == pattern) {
        return true;
    }

    // Wildcard match (*.example.com matches sub.example.com)
    if (pattern.size() > 2 && pattern[0] == '*' && pattern[1] == '.') {
        std::string suffix = pattern.substr(1);  // .example.com
        if (domain.size() > suffix.size()) {
            return domain.substr(domain.size() - suffix.size()) == suffix;
        }
    }

    return false;
}

} // namespace clove::kernel
