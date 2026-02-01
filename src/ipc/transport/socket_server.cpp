#include "ipc/transport/socket_server.hpp"
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>

namespace clove::ipc {

SocketServer::SocketServer(const std::string& socket_path)
    : socket_path_(socket_path) {}

SocketServer::~SocketServer() {
    stop();
}

bool SocketServer::init() {
    // Remove existing socket file
    if (unlink(socket_path_.c_str()) < 0 && errno != ENOENT) {
        // ENOENT means file doesn't exist, which is fine
        if (errno == EACCES || errno == EPERM) {
            spdlog::error("Cannot remove stale socket '{}': Permission denied", socket_path_);
            spdlog::error("This may happen if kernel was previously run as root.");
            spdlog::error("Fix with: sudo rm {}", socket_path_);
            return false;
        }
        // Other errors are non-fatal, we'll try to bind anyway
        spdlog::debug("Could not remove socket file: {}", strerror(errno));
    }

    // Create Unix domain socket
    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        spdlog::error("Failed to create socket: {}", strerror(errno));
        return false;
    }

    // Set non-blocking
    int flags = fcntl(server_fd_, F_GETFL, 0);
    if (fcntl(server_fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
        spdlog::error("Failed to set non-blocking: {}", strerror(errno));
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    // Bind to socket path
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        spdlog::error("Failed to bind socket: {}", strerror(errno));
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    // Listen for connections
    if (listen(server_fd_, 16) < 0) {
        spdlog::error("Failed to listen: {}", strerror(errno));
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    spdlog::info("Socket server listening on {}", socket_path_);
    return true;
}

void SocketServer::set_handler(MessageHandler handler) {
    handler_ = std::move(handler);
}

int SocketServer::accept_connection() {
    struct sockaddr_un client_addr;
    socklen_t client_len = sizeof(client_addr);

    int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            spdlog::error("Failed to accept: {}", strerror(errno));
        }
        return -1;
    }

    // Set client non-blocking
    int flags = fcntl(client_fd, F_GETFL, 0);
    fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);

    // Assign agent ID
    uint32_t agent_id = next_agent_id_++;
    clients_[client_fd] = std::make_unique<ClientConnection>(client_fd, agent_id);

    spdlog::info("Agent {} connected (fd={})", agent_id, client_fd);
    return client_fd;
}

bool SocketServer::handle_client(int client_fd) {
    auto it = clients_.find(client_fd);
    if (it == clients_.end()) {
        return false;
    }

    auto& client = *it->second;
    uint8_t buffer[4096];

    // Read available data
    while (true) {
        ssize_t n = read(client_fd, buffer, sizeof(buffer));
        if (n > 0) {
            client.recv_buffer.insert(client.recv_buffer.end(), buffer, buffer + n);
        } else if (n == 0) {
            // Client disconnected
            spdlog::info("Agent {} disconnected (fd={})", client.agent_id, client_fd);
            return false;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break; // No more data
            }
            spdlog::error("Read error for agent {}: {}", client.agent_id, strerror(errno));
            return false;
        }
    }

    // Process complete messages
    process_messages(client);
    return true;
}

void SocketServer::process_messages(ClientConnection& client) {
    while (true) {
        // Check if we have a complete message
        auto msg_size = Message::get_message_size(
            client.recv_buffer.data(),
            client.recv_buffer.size()
        );

        if (!msg_size || client.recv_buffer.size() < *msg_size) {
            break; // Need more data
        }

        // Parse message
        auto msg = Message::deserialize(
            client.recv_buffer.data(),
            client.recv_buffer.size()
        );

        if (!msg) {
            spdlog::warn("Invalid message from agent {}", client.agent_id);
            // Remove bad data - skip header
            client.recv_buffer.erase(
                client.recv_buffer.begin(),
                client.recv_buffer.begin() + HEADER_SIZE
            );
            continue;
        }

        spdlog::debug("Agent {} -> {} ({}B payload)",
            client.agent_id,
            opcode_to_string(msg->opcode),
            msg->payload.size()
        );

        // Remove processed message from buffer
        client.recv_buffer.erase(
            client.recv_buffer.begin(),
            client.recv_buffer.begin() + *msg_size
        );

        // Call handler and queue response
        if (handler_) {
            // Override message agent_id with actual client ID (client may send 0 initially)
            msg->agent_id = client.agent_id;
            Message response = handler_(*msg);
            response.agent_id = client.agent_id;

            auto serialized = response.serialize();
            client.send_buffer.insert(
                client.send_buffer.end(),
                serialized.begin(),
                serialized.end()
            );
            client.want_write = true;

            spdlog::debug("Agent {} <- {} ({}B payload)",
                client.agent_id,
                opcode_to_string(response.opcode),
                response.payload.size()
            );
        }
    }
}

bool SocketServer::flush_client(int client_fd) {
    auto it = clients_.find(client_fd);
    if (it == clients_.end()) {
        return false;
    }

    auto& client = *it->second;

    while (!client.send_buffer.empty()) {
        ssize_t n = write(client_fd,
            client.send_buffer.data(),
            client.send_buffer.size()
        );

        if (n > 0) {
            client.send_buffer.erase(
                client.send_buffer.begin(),
                client.send_buffer.begin() + n
            );
        } else if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break; // Would block, try later
            }
            spdlog::error("Write error for agent {}: {}", client.agent_id, strerror(errno));
            return false;
        }
    }

    client.want_write = !client.send_buffer.empty();
    return true;
}

bool SocketServer::client_wants_write(int client_fd) const {
    auto it = clients_.find(client_fd);
    if (it == clients_.end()) {
        return false;
    }
    return it->second->want_write;
}

uint32_t SocketServer::remove_client(int client_fd) {
    auto it = clients_.find(client_fd);
    if (it != clients_.end()) {
        uint32_t agent_id = it->second->agent_id;
        close(client_fd);
        clients_.erase(it);
        return agent_id;
    }
    return 0;
}

void SocketServer::stop() {
    // Close all clients
    for (auto& [fd, client] : clients_) {
        close(fd);
    }
    clients_.clear();

    // Close server socket
    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }

    // Remove socket file
    unlink(socket_path_.c_str());
    spdlog::info("Socket server stopped");
}

} // namespace clove::ipc
