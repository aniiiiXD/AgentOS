#pragma once
#include <functional>
#include <unordered_map>
#include <cstdint>

namespace agentos::kernel {

// Event types
enum class EventType : uint32_t {
    READABLE  = 0x001,
    WRITABLE  = 0x004,
    ERROR     = 0x008,
    HANGUP    = 0x010
};

// Event callback: (fd, events) -> void
using EventCallback = std::function<void(int fd, uint32_t events)>;

class Reactor {
public:
    Reactor();
    ~Reactor();

    // Non-copyable
    Reactor(const Reactor&) = delete;
    Reactor& operator=(const Reactor&) = delete;

    // Initialize epoll
    bool init();

    // Add fd to watch (returns true on success)
    bool add(int fd, uint32_t events, EventCallback callback);

    // Modify watched events for fd
    bool modify(int fd, uint32_t events);

    // Remove fd from watch
    bool remove(int fd);

    // Run one iteration of event loop
    // timeout_ms: -1 = block forever, 0 = return immediately
    int poll(int timeout_ms = -1);

    // Run event loop until stopped
    void run();

    // Stop the event loop
    void stop();

    // Check if running
    bool is_running() const { return running_; }

private:
    int epoll_fd_ = -1;
    bool running_ = false;
    std::unordered_map<int, EventCallback> callbacks_;
};

// Helper to combine event flags
inline uint32_t operator|(EventType a, EventType b) {
    return static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
}

inline uint32_t operator|(uint32_t a, EventType b) {
    return a | static_cast<uint32_t>(b);
}

} // namespace agentos::kernel
