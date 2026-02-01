#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include "core/logger.hpp"
#include "kernel/kernel.hpp"
#include <iostream>
#include <thread>
#include <chrono>

// ANSI escape codes
namespace term {
    constexpr const char* RESET     = "\033[0m";
    constexpr const char* BOLD      = "\033[1m";
    constexpr const char* DIM       = "\033[2m";
    constexpr const char* CYAN      = "\033[36m";
    constexpr const char* GREEN     = "\033[32m";
    constexpr const char* YELLOW    = "\033[33m";
    constexpr const char* MAGENTA   = "\033[35m";
    constexpr const char* WHITE     = "\033[37m";
    constexpr const char* BG_BLACK  = "\033[40m";
    constexpr const char* CLEAR_LINE = "\033[2K";
    constexpr const char* HIDE_CURSOR = "\033[?25l";
    constexpr const char* SHOW_CURSOR = "\033[?25h";
}

void print_banner() {
    // Clear and print styled banner
    std::cout << term::CYAN << term::BOLD;
    std::cout << R"(
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║       ██████╗██╗      ██████╗ ██╗   ██╗███████╗           ║
    ║      ██╔════╝██║     ██╔═══██╗██║   ██║██╔════╝           ║
    ║      ██║     ██║     ██║   ██║██║   ██║█████╗             ║
    ║      ██║     ██║     ██║   ██║╚██╗ ██╔╝██╔══╝             ║
    ║      ╚██████╗███████╗╚██████╔╝ ╚████╔╝ ███████╗           ║
    ║       ╚═════╝╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝           ║
    ║                                                           ║
    ║)" << term::RESET << term::DIM << "            ≡  Microkernel for Intelligent Agents  ≡" << term::CYAN << term::BOLD << R"(       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
)" << term::RESET;
}

void print_status_box(const std::string& socket_path, bool sandbox_enabled) {
    std::cout << term::WHITE << term::BOLD;
    std::cout << "\n    ┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "    │" << term::RESET << term::CYAN << "  CLOVE STATUS" << term::RESET
              << term::WHITE << term::BOLD << "                                           │\n";
    std::cout << "    ├─────────────────────────────────────────────────────────┤\n";

    // Version
    std::cout << "    │" << term::RESET << "  Version     " << term::GREEN << "v0.1.0"
              << term::RESET << "                                    " << term::WHITE << term::BOLD << "│\n";

    // Socket
    std::cout << "    │" << term::RESET << "  Socket      " << term::YELLOW << socket_path;
    int padding = 41 - socket_path.length();
    for (int i = 0; i < padding; i++) std::cout << " ";
    std::cout << term::WHITE << term::BOLD << "│\n";

    // Sandbox
    std::cout << "    │" << term::RESET << "  Sandbox     ";
    if (sandbox_enabled) {
        std::cout << term::GREEN << "enabled                                   ";
    } else {
        std::cout << term::YELLOW << "disabled                                  ";
    }
    std::cout << term::WHITE << term::BOLD << "│\n";

    std::cout << "    └─────────────────────────────────────────────────────────┘\n" << term::RESET;
}

void print_startup_sequence() {
    const char* spinner[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
    const char* steps[] = {
        "Initializing reactor",
        "Binding socket server",
        "Loading permissions",
        "Configuring tunnels",
        "Starting agent manager"
    };

    for (int step = 0; step < 5; step++) {
        for (int i = 0; i < 6; i++) {
            std::cout << "\r    " << term::CYAN << spinner[i % 10] << term::RESET
                      << "  " << steps[step] << "..." << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::cout << "\r    " << term::GREEN << "✓" << term::RESET
                  << "  " << steps[step] << "    \n";
    }
}

void print_ready_message(const std::string& socket_path) {
    std::cout << "\n" << term::GREEN << term::BOLD;
    std::cout << "    ══════════════════════════════════════════════════════════\n";
    std::cout << "      CLOVE READY" << term::RESET << term::DIM << "  ·  Press Ctrl+C to shutdown\n";
    std::cout << term::GREEN << term::BOLD;
    std::cout << "    ══════════════════════════════════════════════════════════\n";
    std::cout << term::RESET << "\n";
}

void setup_logging() {
    clove::core::init_logger();
    clove::core::set_log_level(spdlog::level::info);
    spdlog::set_pattern("    %^[%l]%$ %v");
}

int main(int argc, char** argv) {
    // Print banner first
    print_banner();

    // Parse command line args
    clove::kernel::Kernel::Config config;
    if (argc > 1) {
        config.socket_path = argv[1];
    }

    // Show startup animation
    print_startup_sequence();

    // Setup logging after animation
    setup_logging();

    // Create kernel
    clove::kernel::Kernel kernel(config);

    // Initialize
    if (!kernel.init()) {
        std::cout << "\n    " << term::BOLD << "\033[31m✗" << term::RESET
                  << "  Failed to initialize clove\n\n";
        return 1;
    }

    // Print status box
    print_status_box(config.socket_path, config.enable_sandboxing);

    // Ready message
    print_ready_message(config.socket_path);

    // Run (blocks until Ctrl+C)
    kernel.run();

    // Shutdown message
    std::cout << "\n    " << term::YELLOW << "⟳" << term::RESET
              << "  Shutting down gracefully...\n\n";

    return 0;
}
