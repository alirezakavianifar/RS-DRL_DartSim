#!/bin/bash
# Setup script for DARTSim in WSL/Ubuntu

set -e

echo "DARTSim Setup Script for WSL/Ubuntu"
echo "===================================="
echo ""

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install required dependencies
echo "Installing dependencies..."
sudo apt-get install -y libboost-all-dev libyaml-cpp-dev make automake autoconf g++ default-jdk ant wget libtool

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DARTSIM_DIR="$SCRIPT_DIR"

# If running from Windows, the path might be in /mnt
if [ ! -d "$DARTSIM_DIR/src" ]; then
    # Try to find dartsim in common locations
    if [ -d "/mnt/e/projects/dart/dartsim-master" ]; then
        DARTSIM_DIR="/mnt/e/projects/dart/dartsim-master"
    elif [ -d "$HOME/dartsim" ]; then
        DARTSIM_DIR="$HOME/dartsim"
    else
        echo "Warning: Could not find DARTSim source directory"
        echo "Please navigate to the dartsim directory and run:"
        echo "  autoreconf -i"
        echo "  mkdir -p build && cd build"
        echo "  ../configure && make"
        exit 1
    fi
fi

echo "Building DARTSim from: $DARTSIM_DIR"
cd "$DARTSIM_DIR"

# Build DARTSim
echo "Running autoreconf..."
autoreconf -i

echo "Creating build directory..."
mkdir -p build
cd build

echo "Running configure..."
../configure

echo "Building DARTSim (this may take a few minutes)..."
make

echo ""
echo "===================================="
echo "DARTSim build completed successfully!"
echo "===================================="
echo ""
echo "To run examples:"
echo "  cd $DARTSIM_DIR/examples/simple-cpp"
echo "  ../run.sh"
echo ""

