#include "kernel/llm_client.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include <cstdlib>

using json = nlohmann::json;

namespace agentos::kernel {

LLMClient::LLMClient(const LLMConfig& config)
    : config_(config) {
    if (config_.api_key.empty()) {
        config_.api_key = get_api_key_from_env();
    }

    if (config_.api_key.empty()) {
        spdlog::warn("No Gemini API key configured. Set GEMINI_API_KEY environment variable.");
    } else {
        spdlog::info("LLM client initialized (model={})", config_.model);
    }
}

LLMClient::~LLMClient() = default;

bool LLMClient::is_configured() const {
    return !config_.api_key.empty();
}

std::string LLMClient::get_api_key_from_env() {
    const char* key = std::getenv("GEMINI_API_KEY");
    if (key) {
        return std::string(key);
    }

    // Also check GOOGLE_API_KEY as fallback
    key = std::getenv("GOOGLE_API_KEY");
    if (key) {
        return std::string(key);
    }

    return "";
}

LLMResponse LLMClient::complete(const std::string& prompt) {
    std::vector<ChatMessage> messages = {
        {"user", prompt}
    };
    return chat(messages);
}

LLMResponse LLMClient::chat(const std::vector<ChatMessage>& messages) {
    if (!is_configured()) {
        LLMResponse response;
        response.success = false;
        response.error = "API key not configured";
        return response;
    }

    std::string request_body = build_request_json(messages);
    return make_request(request_body);
}

std::string LLMClient::build_request_json(const std::vector<ChatMessage>& messages) {
    json request;

    // Build contents array for Gemini format
    json contents = json::array();

    for (const auto& msg : messages) {
        json content;
        content["role"] = (msg.role == "user") ? "user" : "model";

        json parts = json::array();
        json part;
        part["text"] = msg.content;
        parts.push_back(part);

        content["parts"] = parts;
        contents.push_back(content);
    }

    request["contents"] = contents;

    // Generation config
    json gen_config;
    gen_config["temperature"] = config_.temperature;
    gen_config["maxOutputTokens"] = config_.max_tokens;
    request["generationConfig"] = gen_config;

    return request.dump();
}

LLMResponse LLMClient::parse_response(const std::string& response_body) {
    LLMResponse response;

    try {
        json j = json::parse(response_body);

        // Check for error
        if (j.contains("error")) {
            response.success = false;
            response.error = j["error"]["message"].get<std::string>();
            return response;
        }

        // Extract text from candidates
        if (j.contains("candidates") && !j["candidates"].empty()) {
            auto& candidate = j["candidates"][0];

            if (candidate.contains("content") &&
                candidate["content"].contains("parts") &&
                !candidate["content"]["parts"].empty()) {

                response.content = candidate["content"]["parts"][0]["text"].get<std::string>();
                response.success = true;
            }
        }

        // Extract token usage if available
        if (j.contains("usageMetadata")) {
            auto& usage = j["usageMetadata"];
            if (usage.contains("totalTokenCount")) {
                response.tokens_used = usage["totalTokenCount"].get<int>();
            }
        }

        if (!response.success && response.error.empty()) {
            response.error = "No content in response";
        }

    } catch (const std::exception& e) {
        response.success = false;
        response.error = std::string("JSON parse error: ") + e.what();
        spdlog::error("Failed to parse Gemini response: {}", e.what());
        spdlog::debug("Response body: {}", response_body);
    }

    return response;
}

LLMResponse LLMClient::make_request(const std::string& request_body) {
    LLMResponse response;

    try {
        // Create HTTPS client
        httplib::Client cli("https://" + config_.api_host);
        cli.set_connection_timeout(config_.timeout_seconds);
        cli.set_read_timeout(config_.timeout_seconds);
        cli.set_write_timeout(config_.timeout_seconds);

        // Build URL with API key
        std::string path = "/v1beta/models/" + config_.model + ":generateContent";
        path += "?key=" + config_.api_key;

        spdlog::debug("Calling Gemini API: {}", config_.model);

        // Make POST request
        auto result = cli.Post(path, request_body, "application/json");

        if (!result) {
            response.success = false;
            response.error = "HTTP request failed: " + httplib::to_string(result.error());
            spdlog::error("Gemini API request failed: {}", response.error);
            return response;
        }

        spdlog::debug("Gemini API response: {} ({}B)", result->status, result->body.size());

        if (result->status != 200) {
            response.success = false;
            response.error = "HTTP " + std::to_string(result->status);

            // Try to extract error message from response
            try {
                json j = json::parse(result->body);
                if (j.contains("error") && j["error"].contains("message")) {
                    response.error += ": " + j["error"]["message"].get<std::string>();
                }
            } catch (...) {}

            spdlog::error("Gemini API error: {}", response.error);
            return response;
        }

        return parse_response(result->body);

    } catch (const std::exception& e) {
        response.success = false;
        response.error = std::string("Exception: ") + e.what();
        spdlog::error("Gemini API exception: {}", e.what());
    }

    return response;
}

} // namespace agentos::kernel
