/**
 * AgentOS Permission System
 *
 * Controls what agents can do: file access, command execution,
 * HTTP requests, LLM usage. Uses glob patterns for path matching.
 */
#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include <regex>
#include <nlohmann/json.hpp>

namespace clove::kernel {

// Permission levels (presets)
enum class PermissionLevel {
    UNRESTRICTED,   // Everything allowed (dev mode)
    STANDARD,       // Read/write/exec, no spawn/network
    SANDBOXED,      // Limited paths, no dangerous commands
    READONLY,       // Can only read files, no exec
    MINIMAL         // Can only think (LLM calls)
};

// Agent permissions structure
struct AgentPermissions {
    // Syscall permissions
    bool can_exec = false;
    bool can_read = true;
    bool can_write = true;
    bool can_think = true;
    bool can_spawn = false;      // Spawning other agents (dangerous)
    bool can_http = false;       // HTTP requests

    // Filesystem restrictions
    std::vector<std::string> allowed_read_paths;   // Glob patterns, empty = all
    std::vector<std::string> allowed_write_paths;  // Glob patterns, empty = all
    std::vector<std::string> blocked_paths;        // Always deny these

    // Command restrictions
    std::vector<std::string> allowed_commands;     // Prefixes, empty = all
    std::vector<std::string> blocked_commands;     // Always deny (e.g., "rm -rf", "sudo")

    // Network restrictions
    std::vector<std::string> allowed_domains;      // For HTTP, empty = none
    std::vector<std::string> allowed_http_methods; // For HTTP, empty = none

    // Resource limits
    uint64_t max_llm_tokens = 0;        // 0 = unlimited
    uint32_t max_llm_calls = 0;         // 0 = unlimited
    uint64_t max_exec_time_ms = 30000;  // Max execution time per command

    // Tracking
    uint64_t llm_tokens_used = 0;
    uint32_t llm_calls_made = 0;

    // Create from JSON
    static AgentPermissions from_json(const nlohmann::json& j);

    // Serialize to JSON
    nlohmann::json to_json() const;

    // Create from preset level
    static AgentPermissions from_level(PermissionLevel level);

    // Check if a path is allowed for reading
    bool can_read_path(const std::string& path) const;

    // Check if a path is allowed for writing
    bool can_write_path(const std::string& path) const;

    // Check if a command is allowed
    bool can_execute_command(const std::string& command) const;

    // Check if a domain is allowed for HTTP
    bool can_access_domain(const std::string& domain) const;
    bool can_http_method(const std::string& method) const;

    // Check LLM quota
    bool can_use_llm(uint32_t estimated_tokens = 0) const;

    // Record LLM usage
    void record_llm_usage(uint32_t tokens);
};

// Permission checker utility
class PermissionChecker {
public:
    // Match a path against a glob pattern
    static bool path_matches(const std::string& path, const std::string& pattern);

    // Check if command starts with any allowed prefix
    static bool command_matches(const std::string& command, const std::string& prefix);

    // Extract domain from URL
    static std::string extract_domain(const std::string& url);

    // Check if domain matches pattern (supports wildcards like *.example.com)
    static bool domain_matches(const std::string& domain, const std::string& pattern);
};

// Default blocked paths (security-sensitive)
const std::vector<std::string> DEFAULT_BLOCKED_PATHS = {
    "/etc/shadow",
    "/etc/passwd",
    "~/.ssh/*",
    "~/.gnupg/*",
    "~/.aws/*",
    "~/.config/gcloud/*",
    "*/.env",
    "*/.git/config",
    "*/credentials*",
    "*/secrets*",
    "*/*token*",
    "*/*password*"
};

// Default blocked commands (dangerous)
const std::vector<std::string> DEFAULT_BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf ~",
    "rm -rf /*",
    "sudo",
    "su ",
    "chmod 777",
    "curl | bash",
    "wget | bash",
    "> /dev/sd",
    "dd if=",
    "mkfs",
    ":(){:|:&};:",  // Fork bomb
    "shutdown",
    "reboot",
    "init 0",
    "poweroff"
};

} // namespace clove::kernel
