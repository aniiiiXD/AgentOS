#pragma once
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include "kernel/context.hpp"
#include "kernel/event_bus.hpp"
#include "kernel/module.hpp"
#include "kernel/permissions.hpp"
#include "ipc/protocol.hpp"

namespace clove::worlds {
class World;
}

namespace clove::kernel {

class SyscallRouter;

class AgentSyscalls final : public KernelModule {
public:
    explicit AgentSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    ipc::Message handle_spawn(const ipc::Message& msg);
    ipc::Message handle_kill(const ipc::Message& msg);
    ipc::Message handle_list(const ipc::Message& msg);
    ipc::Message handle_pause(const ipc::Message& msg);
    ipc::Message handle_resume(const ipc::Message& msg);
    KernelContext& context_;
};

class AsyncSyscalls final : public KernelModule {
public:
    explicit AsyncSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    ipc::Message handle_async_poll(const ipc::Message& msg);
    KernelContext& context_;
};

class AuditSyscalls final : public KernelModule {
public:
    explicit AuditSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    ipc::Message handle_get_audit_log(const ipc::Message& msg);
    ipc::Message handle_set_audit_config(const ipc::Message& msg);
    KernelContext& context_;
};

class EventSyscalls final : public KernelModule {
public:
    explicit EventSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
    void emit_event(KernelEventType type, const nlohmann::json& data, uint32_t source_agent_id);
private:
    ipc::Message handle_subscribe(const ipc::Message& msg);
    ipc::Message handle_unsubscribe(const ipc::Message& msg);
    ipc::Message handle_poll_events(const ipc::Message& msg);
    ipc::Message handle_emit(const ipc::Message& msg);
    KernelContext& context_;
};

class ExecSyscalls final : public KernelModule {
public:
    explicit ExecSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    static ipc::Message exec_sync(KernelContext& context, const ipc::Message& msg, const nlohmann::json& j);
    ipc::Message handle_exec(const ipc::Message& msg);
    KernelContext& context_;
};

class FileSyscalls final : public KernelModule {
public:
    explicit FileSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    ipc::Message handle_read(const ipc::Message& msg);
    ipc::Message handle_write(const ipc::Message& msg);
    ipc::Message handle_read_virtual(const ipc::Message& msg, clove::worlds::World* world);
    ipc::Message handle_write_virtual(const ipc::Message& msg, clove::worlds::World* world);
    KernelContext& context_;
};

class IpcSyscalls final : public KernelModule {
public:
    explicit IpcSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    ipc::Message handle_register(const ipc::Message& msg);
    ipc::Message handle_send(const ipc::Message& msg);
    ipc::Message handle_recv(const ipc::Message& msg);
    ipc::Message handle_broadcast(const ipc::Message& msg);
    KernelContext& context_;
};

class LlmSyscalls final : public KernelModule {
public:
    explicit LlmSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    static ipc::Message think_sync(KernelContext& context, const ipc::Message& msg);
    ipc::Message handle_think(const ipc::Message& msg);
    ipc::Message handle_report(const ipc::Message& msg);
    KernelContext& context_;
};

class MetricsSyscalls final : public KernelModule {
public:
    explicit MetricsSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    ipc::Message handle_metrics_system(const ipc::Message& msg);
    ipc::Message handle_metrics_agent(const ipc::Message& msg);
    ipc::Message handle_metrics_all_agents(const ipc::Message& msg);
    ipc::Message handle_metrics_cgroup(const ipc::Message& msg);
    KernelContext& context_;
};

class NetworkSyscalls final : public KernelModule {
public:
    explicit NetworkSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    static ipc::Message http_sync(KernelContext& context, const ipc::Message& msg, const nlohmann::json& j);
    ipc::Message handle_http(const ipc::Message& msg);
    ipc::Message handle_http_virtual(const ipc::Message& msg, clove::worlds::World* world);
    KernelContext& context_;
};

class PermissionSyscalls final : public KernelModule {
public:
    explicit PermissionSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    AgentPermissions& get_agent_permissions(uint32_t agent_id);
    ipc::Message handle_get_perms(const ipc::Message& msg);
    ipc::Message handle_set_perms(const ipc::Message& msg);
    KernelContext& context_;
};

class ReplaySyscalls final : public KernelModule {
public:
    explicit ReplaySyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    ipc::Message handle_record_start(const ipc::Message& msg);
    ipc::Message handle_record_stop(const ipc::Message& msg);
    ipc::Message handle_record_status(const ipc::Message& msg);
    ipc::Message handle_replay_start(const ipc::Message& msg);
    ipc::Message handle_replay_status(const ipc::Message& msg);
    KernelContext& context_;
};

class StateSyscalls final : public KernelModule {
public:
    explicit StateSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    ipc::Message handle_store(const ipc::Message& msg);
    ipc::Message handle_fetch(const ipc::Message& msg);
    ipc::Message handle_delete(const ipc::Message& msg);
    ipc::Message handle_keys(const ipc::Message& msg);
    KernelContext& context_;
};

class TunnelSyscalls final : public KernelModule {
public:
    explicit TunnelSyscalls(KernelContext& context,
                            std::function<ipc::Message(const ipc::Message&)> dispatch)
        : context_(context), dispatch_(std::move(dispatch)) {}
    void register_syscalls(SyscallRouter& router) override;
    void on_tick() override;
private:
    ipc::Message handle_tunnel_connect(const ipc::Message& msg);
    ipc::Message handle_tunnel_disconnect(const ipc::Message& msg);
    ipc::Message handle_tunnel_status(const ipc::Message& msg);
    ipc::Message handle_tunnel_list_remotes(const ipc::Message& msg);
    ipc::Message handle_tunnel_config(const ipc::Message& msg);
    void process_tunnel_events();
    void handle_tunnel_syscall(uint32_t agent_id, uint8_t opcode, const std::vector<uint8_t>& payload);
    KernelContext& context_;
    std::function<ipc::Message(const ipc::Message&)> dispatch_;
};

class WorldSyscalls final : public KernelModule {
public:
    explicit WorldSyscalls(KernelContext& context) : context_(context) {}
    void register_syscalls(SyscallRouter& router) override;
private:
    ipc::Message handle_world_create(const ipc::Message& msg);
    ipc::Message handle_world_destroy(const ipc::Message& msg);
    ipc::Message handle_world_list(const ipc::Message& msg);
    ipc::Message handle_world_join(const ipc::Message& msg);
    ipc::Message handle_world_leave(const ipc::Message& msg);
    ipc::Message handle_world_event(const ipc::Message& msg);
    ipc::Message handle_world_state(const ipc::Message& msg);
    ipc::Message handle_world_snapshot(const ipc::Message& msg);
    ipc::Message handle_world_restore(const ipc::Message& msg);
    KernelContext& context_;
};

} // namespace clove::kernel
