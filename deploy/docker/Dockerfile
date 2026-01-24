# Clove Kernel Docker Image
# Multi-stage build for minimal image size

# Stage 1: Build the kernel
FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    pkg-config \
    curl \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install vcpkg for dependencies
WORKDIR /opt
RUN git clone https://github.com/Microsoft/vcpkg.git && \
    ./vcpkg/bootstrap-vcpkg.sh

# Copy project files
WORKDIR /app
COPY CMakeLists.txt vcpkg.json ./
COPY src/ src/

# Install dependencies via vcpkg and build
RUN /opt/vcpkg/vcpkg install --triplet x64-linux
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake && \
    cmake --build . --config Release

# Stage 2: Runtime image
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for relay/agents
RUN pip3 install --no-cache-dir \
    websockets \
    aiohttp \
    python-dotenv \
    pyyaml

# Create non-root user
RUN useradd -m -s /bin/bash clove

# Copy kernel binary
COPY --from=builder /app/build/clove_kernel /usr/local/bin/

# Copy relay and agent SDK
COPY relay/ /app/relay/
COPY agents/python_sdk/ /app/agents/python_sdk/

# Copy entrypoint
COPY deploy/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create directories for agents and data
RUN mkdir -p /app/agents/scripts /app/data && \
    chown -R clove:clove /app

# Set environment variables
ENV CLOVE_HOME=/app
ENV PYTHONPATH=/app
ENV CLOVE_DATA_DIR=/app/data

# Expose ports
# 9000 - Kernel IPC (Unix socket proxied via TCP for container access)
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep clove_kernel || exit 1

# Run as non-root
USER clove
WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]
CMD ["kernel"]
