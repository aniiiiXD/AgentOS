#include "kernel/reactor.hpp"
#include <spdlog/spdlog.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

namespace agentos::kernel {

Reactor::Reactor() = default;

Reactor::~Reactor() {
    if (epoll_fd_ >= 0) {
        close(epoll_fd_);
    }
}

bool Reactor::init() {
    epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
    if (epoll_fd_ < 0) {
        spdlog::error("Failed to create epoll: {}", strerror(errno));
        return false;
    }
    spdlog::debug("Reactor initialized (epoll_fd={})", epoll_fd_);
    return true;
}

bool Reactor::add(int fd, uint32_t events, EventCallback callback) {
    struct epoll_event ev;
    ev.events = events;
    ev.data.fd = fd;

    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) {
        spdlog::error("Failed to add fd {} to epoll: {}", fd, strerror(errno));
        return false;
    }

    callbacks_[fd] = std::move(callback);
    spdlog::debug("Added fd {} to reactor (events=0x{:x})", fd, events);
    return true;
}

bool Reactor::modify(int fd, uint32_t events) {
    struct epoll_event ev;
    ev.events = events;
    ev.data.fd = fd;

    if (epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev) < 0) {
        spdlog::error("Failed to modify fd {} in epoll: {}", fd, strerror(errno));
        return false;
    }

    spdlog::debug("Modified fd {} in reactor (events=0x{:x})", fd, events);
    return true;
}

bool Reactor::remove(int fd) {
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr) < 0) {
        // ENOENT is ok - fd might already be closed
        if (errno != ENOENT) {
            spdlog::error("Failed to remove fd {} from epoll: {}", fd, strerror(errno));
            return false;
        }
    }

    callbacks_.erase(fd);
    spdlog::debug("Removed fd {} from reactor", fd);
    return true;
}

int Reactor::poll(int timeout_ms) {
    constexpr int MAX_EVENTS = 64;
    struct epoll_event events[MAX_EVENTS];

    int n = epoll_wait(epoll_fd_, events, MAX_EVENTS, timeout_ms);
    if (n < 0) {
        if (errno == EINTR) {
            return 0; // Interrupted, not an error
        }
        spdlog::error("epoll_wait failed: {}", strerror(errno));
        return -1;
    }

    // Process events
    for (int i = 0; i < n; i++) {
        int fd = events[i].data.fd;
        uint32_t ev = events[i].events;

        auto it = callbacks_.find(fd);
        if (it != callbacks_.end()) {
            it->second(fd, ev);
        }
    }

    return n;
}

void Reactor::run() {
    running_ = true;
    spdlog::info("Reactor starting event loop");

    while (running_) {
        int n = poll(100); // 100ms timeout for responsiveness
        if (n < 0) {
            break;
        }
    }

    spdlog::info("Reactor event loop stopped");
}

void Reactor::stop() {
    running_ = false;
}

} // namespace agentos::kernel
