#include "util/logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>

namespace agentos::util {

void init_logger() {
    auto console = spdlog::stdout_color_mt("agentos");
    spdlog::set_default_logger(console);
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
}

void set_log_level(spdlog::level::level_enum level) {
    spdlog::set_level(level);
}

} // namespace agentos::util
