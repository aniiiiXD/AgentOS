cmake_minimum_required(VERSION 3.20)
project(AgentOS VERSION 0.1.0 LANGUAGES CXX)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find vcpkg toolchain
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
endif()

# Dependencies
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(httplib CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

# Optional: io_uring (for modern Linux)
find_package(PkgConfig)
pkg_check_modules(LIBURING liburing)

# Main executable
add_executable(agentos_kernel
    src/main.cpp
    src/kernel/kernel.cpp
    src/kernel/reactor.cpp
    src/kernel/llm_client.cpp
    src/ipc/socket_server.cpp
    src/runtime/sandbox.cpp
    src/runtime/agent_process.cpp
    src/util/logger.cpp
)

target_include_directories(agentos_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(agentos_kernel PRIVATE
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    httplib::httplib
    OpenSSL::SSL
    OpenSSL::Crypto
    pthread
)

# Link io_uring if available
if(LIBURING_FOUND)
    target_compile_definitions(agentos_kernel PRIVATE HAS_IO_URING)
    target_link_libraries(agentos_kernel PRIVATE ${LIBURING_LIBRARIES})
    target_include_directories(agentos_kernel PRIVATE ${LIBURING_INCLUDE_DIRS})
endif()

# Installation
install(TARGETS agentos_kernel DESTINATION bin)
