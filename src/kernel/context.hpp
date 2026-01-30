#pragma once

#include "kernel/config.hpp"

namespace clove::kernel {

class Reactor;
class AuditLogger;
class ExecutionLogger;
class StateStore;
class EventBus;
class AgentMailboxRegistry;
class PermissionsStore;
class AsyncTaskManager;
class LlmQueue;

} // namespace clove::kernel

namespace clove::ipc {
class SocketServer;
} // namespace clove::ipc

namespace clove::runtime {
class AgentManager;
} // namespace clove::runtime

namespace clove::services::tunnel {
class TunnelClient;
} // namespace clove::services::tunnel

namespace clove::metrics {
class MetricsCollector;
} // namespace clove::metrics

namespace clove::worlds {
class WorldEngine;
} // namespace clove::worlds

namespace clove::kernel {

struct KernelContext {
    KernelConfig& config;
    Reactor& reactor;
    ipc::SocketServer& socket_server;
    runtime::AgentManager& agent_manager;
    worlds::WorldEngine& world_engine;
    services::tunnel::TunnelClient& tunnel_client;
    metrics::MetricsCollector& metrics;
    AuditLogger& audit_logger;
    ExecutionLogger& execution_logger;
    StateStore& state_store;
    EventBus& event_bus;
    AgentMailboxRegistry& mailbox_registry;
    PermissionsStore& permissions_store;
    AsyncTaskManager& async_tasks;
    LlmQueue& llm_queue;
};

} // namespace clove::kernel
