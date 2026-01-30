#pragma once
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <future>
#include "services/llm/client.hpp"

namespace clove::kernel {

class LlmQueue {
public:
    explicit LlmQueue(services::llm::LLMClient& client);
    ~LlmQueue();

    LlmQueue(const LlmQueue&) = delete;
    LlmQueue& operator=(const LlmQueue&) = delete;

    std::future<services::llm::LLMResponse> submit(uint32_t agent_id, const std::string& payload);

private:
    struct Request {
        uint32_t agent_id;
        std::string payload;
        std::promise<services::llm::LLMResponse> promise;
    };

    services::llm::LLMClient& client_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<uint32_t, std::deque<Request>> queues_;
    std::deque<uint32_t> round_robin_;
    std::thread worker_;
    bool stopping_ = false;

    void worker_loop();
};

} // namespace clove::kernel
