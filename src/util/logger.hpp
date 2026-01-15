#pragma once
#include <spdlog/spdlog.h>

namespace agentos::util {

// Initialize logging with console output
void init_logger();

// Set log level
void set_log_level(spdlog::level::level_enum level);

} // namespace agentos::util
