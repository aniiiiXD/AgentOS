#!/bin/bash
set -e

# Clove Runtime Entrypoint
# Modes: serve (default), demo, shell

KERNEL_SOCKET="/tmp/clove.sock"

log_info() {
    echo -e "\033[0;32m[clove]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[clove]\033[0m $1"
}

wait_for_kernel() {
    for i in {1..30}; do
        if [ -S "$KERNEL_SOCKET" ]; then
            return 0
        fi
        sleep 1
    done
    log_error "Kernel failed to start (no socket after 30s)"
    return 1
}

# Export API key so kernel subprocess can read it
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"

case "${1:-serve}" in
    serve)
        # Default: run as a runtime service (like postgres)
        log_info "Clove Runtime v0.1.0"
        log_info "Socket: $KERNEL_SOCKET"
        if [ -n "$GEMINI_API_KEY" ]; then
            log_info "LLM:    configured"
        else
            log_info "LLM:    not configured (set GEMINI_API_KEY)"
        fi
        log_info "Starting kernel..."
        exec /usr/local/bin/clove_kernel
        ;;

    demo)
        log_info "Starting Clove kernel..."
        /usr/local/bin/clove_kernel &
        KERNEL_PID=$!

        wait_for_kernel || exit 1
        sleep 1

        log_info "Running fault isolation demo..."
        echo ""
        python3 /app/demos/fault_isolation_demo.py
        echo ""
        log_info "Demo complete."

        kill $KERNEL_PID 2>/dev/null || true
        wait $KERNEL_PID 2>/dev/null || true
        ;;

    shell)
        log_info "Starting Clove kernel..."
        /usr/local/bin/clove_kernel &
        KERNEL_PID=$!

        wait_for_kernel || exit 1

        log_info "Kernel running. SDK available:"
        log_info "  from clove_sdk import CloveClient"
        echo ""
        exec /bin/bash
        ;;

    *)
        exec "$@"
        ;;
esac
