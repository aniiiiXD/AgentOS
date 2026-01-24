# Clove - Standalone Demo Image
# Multi-stage build for the Clove microkernel

# Stage 1: Build the kernel
FROM ubuntu:22.04 AS builder

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

# Install vcpkg
WORKDIR /opt
RUN git clone https://github.com/Microsoft/vcpkg.git && \
    ./vcpkg/bootstrap-vcpkg.sh

# Copy project files
WORKDIR /app
COPY CMakeLists.txt vcpkg.json ./
COPY src/ src/

# Build
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
             -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release

# Stage 2: Runtime
FROM ubuntu:22.04 AS runtime

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libssl3 \
    ca-certificates \
    cgroup-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy kernel binary
COPY --from=builder /app/build/clove_kernel /usr/local/bin/clove_kernel
RUN chmod +x /usr/local/bin/clove_kernel

# Install Python SDK
COPY agents/python_sdk/ /tmp/clove_sdk_src/
RUN pip3 install --no-cache-dir /tmp/clove_sdk_src/ && \
    rm -rf /tmp/clove_sdk_src/

# Copy demo agents
COPY docker/demos/ /app/demos/
COPY agents/examples/cpu_hog_agent.py /app/agents/
COPY agents/examples/memory_hog_agent.py /app/agents/
COPY agents/examples/healthy_agent.py /app/agents/

# Copy entrypoint
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app

# Run as root (needed for namespaces/cgroups)
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
