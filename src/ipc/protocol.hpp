#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <optional>
#include <stdexcept>

namespace agentos::ipc {

// Magic bytes for protocol validation
constexpr uint32_t MAGIC_BYTES = 0x41474E54; // "AGNT" in hex
constexpr size_t HEADER_SIZE = 17;
constexpr size_t MAX_PAYLOAD_SIZE = 1024 * 1024; // 1MB max

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
        case SyscallOp::SYS_NOOP:  return "NOOP";
        case SyscallOp::SYS_THINK: return "THINK";
        case SyscallOp::SYS_EXEC:  return "EXEC";
        case SyscallOp::SYS_READ:  return "READ";
        case SyscallOp::SYS_WRITE: return "WRITE";
        case SyscallOp::SYS_SPAWN: return "SPAWN";
        case SyscallOp::SYS_KILL:  return "KILL";
        case SyscallOp::SYS_LIST:  return "LIST";
        case SyscallOp::SYS_EXIT:  return "EXIT";
        default: return "UNKNOWN";
    }
}

} // namespace agentos::ipc
