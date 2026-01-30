#pragma once
#include <string>
#include <vector>

namespace clove::services::llm {

struct LLMConfig {
    std::string api_key;
    std::string model = "gemini-2.0-flash";
};

struct LLMResponse {
    bool success = false;
    std::string content;
    uint32_t tokens_used = 0;
    std::string error;
};

class LLMClient {
public:
    explicit LLMClient(const LLMConfig& config);
    ~LLMClient();

    LLMClient(const LLMClient&) = delete;
    LLMClient& operator=(const LLMClient&) = delete;

    bool is_configured() const;
    const LLMConfig& config() const { return config_; }

    LLMResponse complete_with_options(const std::string& json_payload);

private:
    LLMConfig config_;
    pid_t subprocess_pid_ = -1;
    int stdin_fd_ = -1;
    int stdout_fd_ = -1;

    bool start_subprocess();
    void stop_subprocess();
    LLMResponse call_subprocess(const std::string& request_json);
    LLMResponse parse_subprocess_response(const std::string& response_json);

    static std::string get_api_key_from_env();
    static std::string get_model_from_env();
};

} // namespace clove::services::llm
