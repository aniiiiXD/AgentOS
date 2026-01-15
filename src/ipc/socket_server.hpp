#pragma once
#include <string>
#include <functional>
#include <vector>
#include <unordered_map>
#include <memory>
#include "ipc/protocol.hpp"

namespace agentos::ipc {

// Client connection state
struct ClientConnection {
    int fd;
    uint32_t agent_id;
    std::vector<uint8_t> recv_buffer;
    std::vector<uint8_t> send_buffer;
    bool want_write = false;

    explicit ClientConnection(int fd, uint32_t id) : fd(fd), agent_id(id) {}
};

// Message handler callback type
using MessageHandler = std::function<Message(const Message&)>;

class SocketServer {
public:
    explicit SocketServer(const std::string& socket_path);
    ~SocketServer();

    // Non-copyable
    SocketServer(const SocketServer&) = delete;
    SocketServer& operator=(const SocketServer&) = delete;

    // Initialize and bind socket
    bool init();

    // Set message handler
    void set_handler(MessageHandler handler);

    // Get server fd for event loop
    int get_server_fd() const { return server_fd_; }

    // Accept new connection, returns client fd
    int accept_connection();

    // Handle client data (read/process/respond)
    // Returns false if client disconnected
    bool handle_client(int client_fd);

    // Send pending data to client
    bool flush_client(int client_fd);

    // Check if client wants to write
    bool client_wants_write(int client_fd) const;

    // Remove client
    void remove_client(int client_fd);

    // Cleanup
    void stop();

    // Get socket path
    const std::string& socket_path() const { return socket_path_; }

private:
    std::string socket_path_;
    int server_fd_ = -1;
    uint32_t next_agent_id_ = 1;
    std::unordered_map<int, std::unique_ptr<ClientConnection>> clients_;
    MessageHandler handler_;

    // Process complete messages in client buffer
    void process_messages(ClientConnection& client);
};

} // namespace agentos::ipc
