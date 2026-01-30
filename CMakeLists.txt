cmake_minimum_required(VERSION 3.20)
project(Clove VERSION 0.1.0 LANGUAGES CXX)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Find vcpkg toolchain
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
endif()

# Dependencies - try to find system packages first, fetch if not found
find_package(fmt QUIET)
if(NOT fmt_FOUND)
    FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 10.2.1
    )
    FetchContent_MakeAvailable(fmt)
endif()

find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.13.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_MakeAvailable(json)
endif()
# OpenSSL for potential future HTTPS syscalls
find_package(OpenSSL REQUIRED)

# Optional: io_uring (for modern Linux)
find_package(PkgConfig)
pkg_check_modules(LIBURING liburing)

# Main executable
add_executable(clove_kernel
    src/main.cpp
    src/kernel/kernel.cpp
    src/kernel/reactor.cpp
    src/kernel/permissions.cpp
    src/kernel/audit_log.cpp
    src/kernel/execution_log.cpp
    src/kernel/syscall_router.cpp
    src/kernel/event_bus.cpp
    src/kernel/state_store.cpp
    src/kernel/ipc_mailbox.cpp
    src/kernel/permissions_store.cpp
    src/kernel/async_task_manager.cpp
    src/kernel/llm_queue.cpp
    src/kernel/syscalls/async.cpp
    src/kernel/syscalls/agents.cpp
    src/kernel/syscalls/audit.cpp
    src/kernel/syscalls/events.cpp
    src/kernel/syscalls/exec.cpp
    src/kernel/syscalls/fs.cpp
    src/kernel/syscalls/ipc.cpp
    src/kernel/syscalls/llm.cpp
    src/kernel/syscalls/metrics.cpp
    src/kernel/syscalls/network.cpp
    src/kernel/syscalls/permissions.cpp
    src/kernel/syscalls/replay.cpp
    src/kernel/syscalls/state.cpp
    src/kernel/syscalls/tunnel.cpp
    src/kernel/syscalls/worlds.cpp
    src/ipc/transport/socket_server.cpp
    src/runtime/sandbox/sandbox.cpp
    src/runtime/agent/process.cpp
    src/runtime/agent/manager.cpp
    src/runtime/agent/types.cpp
    src/services/llm/client.cpp
    src/services/tunnel/client.cpp
    src/metrics/metrics.cpp
    src/worlds/virtual_fs.cpp
    src/worlds/world_engine.cpp
    src/core/logger.cpp
    src/core/config.cpp
    src/core/paths.cpp
)

target_include_directories(clove_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(clove_kernel PRIVATE
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    OpenSSL::SSL
    OpenSSL::Crypto
    pthread
)

# Link io_uring if available
if(LIBURING_FOUND)
    target_compile_definitions(clove_kernel PRIVATE HAS_IO_URING)
    target_link_libraries(clove_kernel PRIVATE ${LIBURING_LIBRARIES})
    target_include_directories(clove_kernel PRIVATE ${LIBURING_INCLUDE_DIRS})
endif()

# Installation
install(TARGETS clove_kernel DESTINATION bin)
