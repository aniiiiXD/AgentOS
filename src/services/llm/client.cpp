#include "services/llm/client.hpp"
#include "core/paths.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <filesystem>
#include <cstring>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace clove::services::llm {

LLMClient::LLMClient(const LLMConfig& config)
    : config_(config) {
    if (config_.api_key.empty()) {
        config_.api_key = get_api_key_from_env();
    }
    if (config_.model.empty()) {
        config_.model = get_model_from_env();
    }
}

LLMClient::~LLMClient() {
    stop_subprocess();
}

bool LLMClient::is_configured() const {
    return !config_.api_key.empty();
}

std::string LLMClient::get_api_key_from_env() {
    const char* key = std::getenv("GEMINI_API_KEY");
    if (!key) {
        key = std::getenv("GOOGLE_API_KEY");
    }
    return key ? std::string(key) : "";
}

std::string LLMClient::get_model_from_env() {
    const char* model = std::getenv("GEMINI_MODEL");
    return model ? std::string(model) : "";
}

bool LLMClient::start_subprocess() {
    if (subprocess_pid_ > 0) {
        return true;
    }

    auto script_path = clove::core::paths::find_relative("agents/llm_service/llm_service.py");
    if (!script_path || !fs::exists(*script_path)) {
        spdlog::error("Could not find agents/llm_service/llm_service.py");
        return false;
    }

    int stdin_pipe[2];
    int stdout_pipe[2];
    if (pipe(stdin_pipe) < 0 || pipe(stdout_pipe) < 0) {
        spdlog::error("Failed to create pipes for LLM subprocess");
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        spdlog::error("Failed to fork LLM subprocess");
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        return false;
    }

    if (pid == 0) {
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);

        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        if (!config_.api_key.empty()) {
            setenv("GEMINI_API_KEY", config_.api_key.c_str(), 1);
        }
        if (!config_.model.empty()) {
            setenv("GEMINI_MODEL", config_.model.c_str(), 1);
        }

        execlp("python3", "python3", script_path->c_str(), nullptr);
        _exit(1);
    }

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    subprocess_pid_ = pid;
    stdin_fd_ = stdin_pipe[1];
    stdout_fd_ = stdout_pipe[0];

    return true;
}

void LLMClient::stop_subprocess() {
    if (subprocess_pid_ <= 0) {
        return;
    }

    kill(subprocess_pid_, SIGTERM);
    int status = 0;
    waitpid(subprocess_pid_, &status, 0);

    if (stdin_fd_ >= 0) close(stdin_fd_);
    if (stdout_fd_ >= 0) close(stdout_fd_);

    subprocess_pid_ = -1;
    stdin_fd_ = -1;
    stdout_fd_ = -1;
}

LLMResponse LLMClient::parse_subprocess_response(const std::string& response_json) {
    LLMResponse result;
    try {
        auto j = json::parse(response_json);
        result.success = j.value("success", false);
        result.content = j.value("content", "");
        result.tokens_used = j.value("tokens", 0);
        result.error = j.value("error", "");
    } catch (const std::exception& e) {
        result.success = false;
        result.error = std::string("invalid response: ") + e.what();
    }
    return result;
}

LLMResponse LLMClient::call_subprocess(const std::string& request_json) {
    if (!start_subprocess()) {
        return {false, "", 0, "LLM subprocess unavailable"};
    }

    std::string line = request_json;
    if (line.empty() || line.back() != '\n') {
        line.push_back('\n');
    }

    ssize_t wrote = write(stdin_fd_, line.data(), line.size());
    if (wrote < 0) {
        return {false, "", 0, "failed to write to LLM subprocess"};
    }

    std::string response;
    char buf[4096];
    while (true) {
        ssize_t n = read(stdout_fd_, buf, sizeof(buf));
        if (n <= 0) {
            break;
        }
        response.append(buf, buf + n);
        if (response.find('\n') != std::string::npos) {
            break;
        }
    }

    auto newline = response.find('\n');
    if (newline != std::string::npos) {
        response = response.substr(0, newline);
    }

    return parse_subprocess_response(response);
}

LLMResponse LLMClient::complete_with_options(const std::string& json_payload) {
    if (!is_configured()) {
        return {false, "", 0, "LLM not configured (set GEMINI_API_KEY)"};
    }
    return call_subprocess(json_payload);
}

} // namespace clove::services::llm
