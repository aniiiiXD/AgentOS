/**
 * AgentOS Wire Protocol
 *
 * Binary protocol for kernel <-> agent communication.
 * Header: 17 bytes (magic + agent_id + opcode + payload_size)
 * See docs/syscalls.md for full syscall reference.
 */
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <optional>
#include <stdexcept>

namespace clove::ipc {

// Magic bytes for protocol validation
constexpr uint32_t MAGIC_BYTES = 0x41474E54; // "AGNT" in hex
constexpr size_t HEADER_SIZE = 17;
constexpr size_t MAX_PAYLOAD_SIZE = 1024 * 1024; // 1MB max
constexpr uint16_t PROTOCOL_VERSION = 1;

// Syscall operations
enum class SyscallOp : uint8_t {
    SYS_NOOP   = 0x00,  // For testing / echo
    SYS_THINK  = 0x01,  // Send prompt to LLM
    SYS_EXEC   = 0x02,  // Execute shell command
    SYS_READ   = 0x03,  // Read file
    SYS_WRITE  = 0x04,  // Write file
    SYS_SPAWN  = 0x10,  // Spawn a sandboxed agent
    SYS_KILL   = 0x11,  // Kill an agent
    SYS_LIST   = 0x12,  // List running agents
    SYS_PAUSE  = 0x14,  // Pause an agent (SIGSTOP)
    SYS_RESUME = 0x15,  // Resume a paused agent (SIGCONT)
    // IPC - Inter-Agent Communication
    SYS_SEND      = 0x20,  // Send message to another agent
    SYS_RECV      = 0x21,  // Receive pending messages
    SYS_BROADCAST = 0x22,  // Broadcast message to all agents
    SYS_REGISTER  = 0x23,  // Register agent name
    // State Store
    SYS_STORE     = 0x30,  // Store key-value pair
    SYS_FETCH     = 0x31,  // Retrieve value by key
    SYS_DELETE    = 0x32,  // Delete a key
    SYS_KEYS      = 0x33,  // List keys with optional prefix
    // Permissions
    SYS_GET_PERMS = 0x40,  // Get own permissions
    SYS_SET_PERMS = 0x41,  // Set agent permissions (privileged)
    // Network
    SYS_HTTP      = 0x50,  // Make HTTP request
    // Events (Pub/Sub)
    SYS_SUBSCRIBE   = 0x60,  // Subscribe to event types
    SYS_UNSUBSCRIBE = 0x61,  // Unsubscribe from events
    SYS_POLL_EVENTS = 0x62,  // Get pending events
    SYS_EMIT        = 0x63,  // Emit custom event
    // World Simulation
    SYS_WORLD_CREATE   = 0xA0,  // Create world from config
    SYS_WORLD_DESTROY  = 0xA1,  // Destroy world
    SYS_WORLD_LIST     = 0xA2,  // List active worlds
    SYS_WORLD_JOIN     = 0xA3,  // Join agent to world
    SYS_WORLD_LEAVE    = 0xA4,  // Remove agent from world
    SYS_WORLD_EVENT    = 0xA5,  // Inject chaos event
    SYS_WORLD_STATE    = 0xA6,  // Get world metrics
    SYS_WORLD_SNAPSHOT = 0xA7,  // Save world state
    SYS_WORLD_RESTORE  = 0xA8,  // Restore from snapshot
    // Remote Connectivity (Tunnel)
    SYS_TUNNEL_CONNECT    = 0xB0,  // Connect kernel to relay server
    SYS_TUNNEL_DISCONNECT = 0xB1,  // Disconnect from relay
    SYS_TUNNEL_STATUS     = 0xB2,  // Get tunnel connection status
    SYS_TUNNEL_LIST_REMOTES = 0xB3,  // List connected remote agents
    SYS_TUNNEL_CONFIG     = 0xB4,  // Configure tunnel settings
    // Metrics
    SYS_METRICS_SYSTEM    = 0xC0,  // Get system-wide metrics (CPU, memory, etc.)
    SYS_METRICS_AGENT     = 0xC1,  // Get metrics for specific agent
    SYS_METRICS_ALL_AGENTS = 0xC2, // Get metrics for all agents
    SYS_METRICS_CGROUP    = 0xC3,  // Get cgroup metrics for sandboxed agent
    // Audit Logging
    SYS_GET_AUDIT_LOG     = 0x76,  // Get audit log entries
    SYS_SET_AUDIT_CONFIG  = 0x77,  // Configure audit logging
    // Execution Recording & Replay
    SYS_RECORD_START   = 0x70,  // Start recording execution
    SYS_RECORD_STOP    = 0x71,  // Stop recording
    SYS_RECORD_STATUS  = 0x72,  // Get recording status
    SYS_REPLAY_START   = 0x73,  // Start replay
    SYS_REPLAY_STATUS  = 0x74,  // Get replay status
    // Async Results
    SYS_ASYNC_POLL     = 0x80,  // Poll for async syscall results
    // Kernel info / capabilities
    SYS_LLM_REPORT     = 0xF0,  // Report SDK LLM usage to kernel
    SYS_HELLO          = 0xFE,  // Handshake / capability query
    SYS_EXIT   = 0xFF   // Graceful shutdown
};

// Response status codes
enum class StatusCode : uint8_t {
    OK          = 0x00,
    ERROR       = 0x01,
    INVALID_MSG = 0x02,
    NOT_FOUND   = 0x03,
    TIMEOUT     = 0x04
};

// Wire protocol header (17 bytes, packed)
struct __attribute__((packed)) MessageHeader {
    uint32_t magic;         // Must be MAGIC_BYTES
    uint32_t agent_id;      // Unique agent identifier
    SyscallOp opcode;       // What operation to perform
    uint64_t payload_size;  // Bytes following this header
};

static_assert(sizeof(MessageHeader) == HEADER_SIZE, "Header size mismatch");

// Application-level message
struct Message {
    uint32_t agent_id;
    SyscallOp opcode;
    std::vector<uint8_t> payload;

    Message() : agent_id(0), opcode(SyscallOp::SYS_NOOP) {}

    Message(uint32_t id, SyscallOp op, const std::vector<uint8_t>& data = {})
        : agent_id(id), opcode(op), payload(data) {}

    Message(uint32_t id, SyscallOp op, const std::string& data)
        : agent_id(id), opcode(op), payload(data.begin(), data.end()) {}

    // Get payload as string
    std::string payload_str() const {
        return std::string(payload.begin(), payload.end());
    }

    // Serialize message to wire format
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buffer(HEADER_SIZE + payload.size());

        MessageHeader header;
        header.magic = MAGIC_BYTES;
        header.agent_id = agent_id;
        header.opcode = opcode;
        header.payload_size = payload.size();

        std::memcpy(buffer.data(), &header, HEADER_SIZE);
        if (!payload.empty()) {
            std::memcpy(buffer.data() + HEADER_SIZE, payload.data(), payload.size());
        }

        return buffer;
    }

    // Deserialize message from wire format
    static std::optional<Message> deserialize(const uint8_t* data, size_t len) {
        if (len < HEADER_SIZE) {
            return std::nullopt;
        }

        MessageHeader header;
        std::memcpy(&header, data, HEADER_SIZE);

        // Validate magic bytes
        if (header.magic != MAGIC_BYTES) {
            return std::nullopt;
        }

        // Validate payload size
        if (header.payload_size > MAX_PAYLOAD_SIZE) {
            return std::nullopt;
        }

        // Check we have complete message
        if (len < HEADER_SIZE + header.payload_size) {
            return std::nullopt;
        }

        Message msg;
        msg.agent_id = header.agent_id;
        msg.opcode = header.opcode;

        if (header.payload_size > 0) {
            msg.payload.resize(header.payload_size);
            std::memcpy(msg.payload.data(), data + HEADER_SIZE, header.payload_size);
        }

        return msg;
    }

    // Helper to get total message size from header
    static std::optional<size_t> get_message_size(const uint8_t* data, size_t len) {
        if (len < HEADER_SIZE) {
            return std::nullopt;
        }

        MessageHeader header;
        std::memcpy(&header, data, HEADER_SIZE);

        if (header.magic != MAGIC_BYTES) {
            return std::nullopt;
        }

        if (header.payload_size > MAX_PAYLOAD_SIZE) {
            return std::nullopt;
        }

        return HEADER_SIZE + header.payload_size;
    }
};

// Convert opcode to string for logging
inline const char* opcode_to_string(SyscallOp op) {
    switch (op) {
        case SyscallOp::SYS_NOOP:      return "NOOP";
        case SyscallOp::SYS_THINK:     return "THINK";
        case SyscallOp::SYS_EXEC:      return "EXEC";
        case SyscallOp::SYS_READ:      return "READ";
        case SyscallOp::SYS_WRITE:     return "WRITE";
        case SyscallOp::SYS_SPAWN:     return "SPAWN";
        case SyscallOp::SYS_KILL:      return "KILL";
        case SyscallOp::SYS_LIST:      return "LIST";
        case SyscallOp::SYS_PAUSE:     return "PAUSE";
        case SyscallOp::SYS_RESUME:    return "RESUME";
        case SyscallOp::SYS_SEND:      return "SEND";
        case SyscallOp::SYS_RECV:      return "RECV";
        case SyscallOp::SYS_BROADCAST: return "BROADCAST";
        case SyscallOp::SYS_REGISTER:  return "REGISTER";
        case SyscallOp::SYS_STORE:     return "STORE";
        case SyscallOp::SYS_FETCH:     return "FETCH";
        case SyscallOp::SYS_DELETE:    return "DELETE";
        case SyscallOp::SYS_KEYS:      return "KEYS";
        case SyscallOp::SYS_GET_PERMS: return "GET_PERMS";
        case SyscallOp::SYS_SET_PERMS: return "SET_PERMS";
        case SyscallOp::SYS_HTTP:      return "HTTP";
        case SyscallOp::SYS_SUBSCRIBE:   return "SUBSCRIBE";
        case SyscallOp::SYS_UNSUBSCRIBE: return "UNSUBSCRIBE";
        case SyscallOp::SYS_POLL_EVENTS: return "POLL_EVENTS";
        case SyscallOp::SYS_EMIT:        return "EMIT";
        case SyscallOp::SYS_WORLD_CREATE:   return "WORLD_CREATE";
        case SyscallOp::SYS_WORLD_DESTROY:  return "WORLD_DESTROY";
        case SyscallOp::SYS_WORLD_LIST:     return "WORLD_LIST";
        case SyscallOp::SYS_WORLD_JOIN:     return "WORLD_JOIN";
        case SyscallOp::SYS_WORLD_LEAVE:    return "WORLD_LEAVE";
        case SyscallOp::SYS_WORLD_EVENT:    return "WORLD_EVENT";
        case SyscallOp::SYS_WORLD_STATE:    return "WORLD_STATE";
        case SyscallOp::SYS_WORLD_SNAPSHOT: return "WORLD_SNAPSHOT";
        case SyscallOp::SYS_WORLD_RESTORE:  return "WORLD_RESTORE";
        case SyscallOp::SYS_TUNNEL_CONNECT:    return "TUNNEL_CONNECT";
        case SyscallOp::SYS_TUNNEL_DISCONNECT: return "TUNNEL_DISCONNECT";
        case SyscallOp::SYS_TUNNEL_STATUS:     return "TUNNEL_STATUS";
        case SyscallOp::SYS_TUNNEL_LIST_REMOTES: return "TUNNEL_LIST_REMOTES";
        case SyscallOp::SYS_TUNNEL_CONFIG:     return "TUNNEL_CONFIG";
        case SyscallOp::SYS_METRICS_SYSTEM:    return "METRICS_SYSTEM";
        case SyscallOp::SYS_METRICS_AGENT:     return "METRICS_AGENT";
        case SyscallOp::SYS_METRICS_ALL_AGENTS: return "METRICS_ALL_AGENTS";
        case SyscallOp::SYS_METRICS_CGROUP:    return "METRICS_CGROUP";
        case SyscallOp::SYS_GET_AUDIT_LOG:     return "GET_AUDIT_LOG";
        case SyscallOp::SYS_SET_AUDIT_CONFIG:  return "SET_AUDIT_CONFIG";
        case SyscallOp::SYS_RECORD_START:   return "RECORD_START";
        case SyscallOp::SYS_RECORD_STOP:    return "RECORD_STOP";
        case SyscallOp::SYS_RECORD_STATUS:  return "RECORD_STATUS";
        case SyscallOp::SYS_REPLAY_START:   return "REPLAY_START";
        case SyscallOp::SYS_REPLAY_STATUS:  return "REPLAY_STATUS";
        case SyscallOp::SYS_ASYNC_POLL:  return "ASYNC_POLL";
        case SyscallOp::SYS_LLM_REPORT:  return "LLM_REPORT";
        case SyscallOp::SYS_HELLO:       return "HELLO";
        case SyscallOp::SYS_EXIT:      return "EXIT";
        default: return "UNKNOWN";
    }
}

} // namespace clove::ipc
