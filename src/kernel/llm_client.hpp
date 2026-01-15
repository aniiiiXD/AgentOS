#pragma once
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <memory>

namespace agentos::kernel {

// LLM configuration
struct LLMConfig {
    std::string api_key;                                    // Gemini API key
    std::string model = "gemini-2.0-flash";                 // Model to use
    std::string api_host = "generativelanguage.googleapis.com";
    int timeout_seconds = 30;
    float temperature = 0.7f;
    int max_tokens = 1024;
};

// Chat message
struct ChatMessage {
    std::string role;    // "user" or "model"
    std::string content;
};

// LLM response
struct LLMResponse {
    bool success = false;
    std::string content;
    std::string error;
    int tokens_used = 0;
};

// Callback for streaming responses (future use)
using StreamCallback = std::function<void(const std::string& chunk)>;

class LLMClient {
public:
    explicit LLMClient(const LLMConfig& config);
    ~LLMClient();

    // Non-copyable
    LLMClient(const LLMClient&) = delete;
    LLMClient& operator=(const LLMClient&) = delete;

    // Check if configured (has API key)
    bool is_configured() const;

    // Simple completion
    LLMResponse complete(const std::string& prompt);

    // Chat completion with history
    LLMResponse chat(const std::vector<ChatMessage>& messages);

    // Get/set config
    const LLMConfig& config() const { return config_; }
    void set_api_key(const std::string& key) { config_.api_key = key; }
    void set_model(const std::string& model) { config_.model = model; }

    // Load API key from environment
    static std::string get_api_key_from_env();

private:
    LLMConfig config_;

    // Build request JSON for Gemini API
    std::string build_request_json(const std::vector<ChatMessage>& messages);

    // Parse response JSON
    LLMResponse parse_response(const std::string& response_body);

    // Make HTTP request
    LLMResponse make_request(const std::string& request_body);
};

} // namespace agentos::kernel
