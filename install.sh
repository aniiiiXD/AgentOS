#!/bin/bash
set -e

# Clove Installer
# Downloads the latest clove_kernel binary and installs the Python SDK.

REPO="anixd/clove"
INSTALL_DIR="/usr/local/bin"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[clove]${NC} $1"; }
warn() { echo -e "${YELLOW}[clove]${NC} $1"; }
error() { echo -e "${RED}[clove]${NC} $1"; exit 1; }

# Check platform
if [ "$(uname -s)" != "Linux" ]; then
    error "Clove currently only supports Linux."
fi

if [ "$(uname -m)" != "x86_64" ]; then
    error "Clove currently only supports x86_64."
fi

info "Installing Clove..."

# Determine latest release
info "Fetching latest release..."
LATEST=$(curl -s "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$LATEST" ]; then
    warn "No release found. Installing from source..."

    # Check build dependencies
    for cmd in cmake g++ git; do
        if ! command -v $cmd &> /dev/null; then
            error "$cmd is required to build from source. Install it and retry."
        fi
    done

    TMPDIR=$(mktemp -d)
    trap "rm -rf $TMPDIR" EXIT

    info "Cloning repository..."
    git clone --depth 1 "https://github.com/${REPO}.git" "$TMPDIR/clove"

    info "Building kernel..."
    cd "$TMPDIR/clove"
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)

    info "Installing binary..."
    sudo cp clove_kernel "$INSTALL_DIR/clove_kernel"
    sudo chmod +x "$INSTALL_DIR/clove_kernel"

    # Install SDK
    info "Installing Python SDK..."
    pip3 install "$TMPDIR/clove/agents/python_sdk/" 2>/dev/null || \
        pip3 install --user "$TMPDIR/clove/agents/python_sdk/"

else
    info "Latest release: $LATEST"

    # Download binary
    DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${LATEST}/clove_kernel"
    info "Downloading clove_kernel..."

    sudo curl -fSL "$DOWNLOAD_URL" -o "$INSTALL_DIR/clove_kernel"
    sudo chmod +x "$INSTALL_DIR/clove_kernel"

    # Install SDK from git
    info "Installing Python SDK..."
    pip3 install "git+https://github.com/${REPO}.git#subdirectory=agents/python_sdk" 2>/dev/null || \
        pip3 install --user "git+https://github.com/${REPO}.git#subdirectory=agents/python_sdk"
fi

info "Installation complete!"
echo ""
echo "  Usage:"
echo "    clove_kernel              # Start the kernel"
echo "    python3 -c 'from clove_sdk import CloveClient'  # Verify SDK"
echo ""
echo "  Quick start:"
echo "    clove_kernel &"
echo "    python3 -c \""
echo "      from clove_sdk import CloveClient"
echo "      with CloveClient() as c:"
echo "          print(c.echo('Hello Clove!'))"
echo "    \""
echo ""
