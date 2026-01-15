#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "kernel/kernel.hpp"

void setup_logging() {
    auto console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
}

int main(int argc, char** argv) {
    setup_logging();

    spdlog::info("=================================");
    spdlog::info("  AgentOS Kernel v0.1.0");
    spdlog::info("  Phase 1: Echo Server");
    spdlog::info("=================================");

    // Parse command line args
    agentos::kernel::Kernel::Config config;
    if (argc > 1) {
        config.socket_path = argv[1];
    }

    // Create and initialize kernel
    agentos::kernel::Kernel kernel(config);

    if (!kernel.init()) {
        spdlog::error("Failed to initialize kernel");
        return 1;
    }

    // Run (blocks until Ctrl+C)
    kernel.run();

    return 0;
}
