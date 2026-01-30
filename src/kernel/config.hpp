#pragma once
#include <string>

namespace clove::kernel {

// Kernel configuration
struct KernelConfig {
    std::string socket_path = "/tmp/clove.sock";
    bool enable_sandboxing = true;
    std::string gemini_api_key;          // Gemini API key (or from env)
    std::string llm_model = "gemini-2.0-flash";
    // Tunnel configuration
    std::string relay_url;               // Relay server URL (ws://...)
    std::string machine_id;              // This machine's ID
    std::string machine_token;           // Authentication token
    bool tunnel_auto_connect = false;    // Auto-connect on startup
};

} // namespace clove::kernel
