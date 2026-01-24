#!/bin/bash
set -e

# Clove Container Entrypoint
# Handles different run modes and connects to relay server

# Default values
MACHINE_ID="${MACHINE_ID:-docker-kernel-$(hostname | cut -c1-8)}"
MACHINE_TOKEN="${MACHINE_TOKEN:-}"
RELAY_URL="${RELAY_URL:-ws://host.docker.internal:8765}"
KERNEL_SOCKET="${KERNEL_SOCKET:-/tmp/clove.sock}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Start the kernel
start_kernel() {
    log_info "Starting Clove Kernel..."
    log_info "  Machine ID: $MACHINE_ID"
    log_info "  Relay URL: $RELAY_URL"

    # Start kernel in background
    /usr/local/bin/clove_kernel &
    KERNEL_PID=$!

    # Wait for kernel socket
    log_info "Waiting for kernel socket..."
    for i in {1..30}; do
        if [ -S "$KERNEL_SOCKET" ]; then
            log_info "Kernel started successfully (PID: $KERNEL_PID)"
            break
        fi
        sleep 1
    done

    if [ ! -S "$KERNEL_SOCKET" ]; then
        log_error "Kernel failed to start (no socket after 30s)"
        exit 1
    fi

    echo $KERNEL_PID
}

# Start the tunnel client (connects kernel to relay)
start_tunnel() {
    log_info "Starting relay tunnel..."

    # Set environment for tunnel client
    export MACHINE_ID
    export MACHINE_TOKEN
    export RELAY_URL

    python3 /app/scripts/tunnel_client.py &
    TUNNEL_PID=$!

    log_info "Tunnel started (PID: $TUNNEL_PID)"
    echo $TUNNEL_PID
}

# Graceful shutdown
shutdown() {
    log_info "Shutting down..."

    if [ -n "$TUNNEL_PID" ]; then
        kill $TUNNEL_PID 2>/dev/null || true
    fi

    if [ -n "$KERNEL_PID" ]; then
        kill $KERNEL_PID 2>/dev/null || true
    fi

    exit 0
}

trap shutdown SIGTERM SIGINT

case "${1:-kernel}" in
    kernel)
        # Default: run kernel + tunnel
        KERNEL_PID=$(start_kernel)

        # Give kernel time to initialize
        sleep 2

        # Start tunnel if relay URL is provided
        if [ -n "$RELAY_URL" ] && [ "$RELAY_URL" != "none" ]; then
            TUNNEL_PID=$(start_tunnel)
        else
            log_info "No relay URL specified, running in standalone mode"
        fi

        # Wait for kernel process
        wait $KERNEL_PID
        ;;

    relay)
        # Run relay server (for local development)
        log_info "Starting Relay Server..."
        cd /app/relay
        python3 relay_server.py --host 0.0.0.0 --port 8765
        ;;

    tunnel)
        # Run tunnel client only (kernel running elsewhere)
        start_tunnel
        wait
        ;;

    agent)
        # Run an agent script
        shift
        if [ -z "$1" ]; then
            log_error "Usage: agent <script.py> [args...]"
            exit 1
        fi

        log_info "Running agent: $1"
        cd /app
        python3 "$@"
        ;;

    shell|bash)
        # Interactive shell
        exec /bin/bash
        ;;

    *)
        # Run arbitrary command
        exec "$@"
        ;;
esac
