#include "kernel/llm_queue.hpp"

namespace clove::kernel {

LlmQueue::LlmQueue(services::llm::LLMClient& client)
    : client_(client) {
    worker_ = std::thread(&LlmQueue::worker_loop, this);
}

LlmQueue::~LlmQueue() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

std::future<services::llm::LLMResponse> LlmQueue::submit(uint32_t agent_id, const std::string& payload) {
    Request req{agent_id, payload, std::promise<services::llm::LLMResponse>()};
    auto future = req.promise.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& q = queues_[agent_id];
        bool was_empty = q.empty();
        q.push_back(std::move(req));
        if (was_empty) {
            round_robin_.push_back(agent_id);
        }
    }
    cv_.notify_one();
    return future;
}

void LlmQueue::worker_loop() {
    while (true) {
        Request req;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return stopping_ || !round_robin_.empty(); });
            if (stopping_) {
                break;
            }

            uint32_t agent_id = round_robin_.front();
            round_robin_.pop_front();

            auto& q = queues_[agent_id];
            req = std::move(q.front());
            q.pop_front();

            if (!q.empty()) {
                round_robin_.push_back(agent_id);
            } else {
                queues_.erase(agent_id);
            }
        }

        auto result = client_.complete_with_options(req.payload);
        req.promise.set_value(result);
    }
}

} // namespace clove::kernel
