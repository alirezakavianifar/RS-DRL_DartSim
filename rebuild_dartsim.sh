#!/bin/bash
# Rebuild DARTSim with the improved connection handling

echo "Rebuilding DARTSim with improved connection handling..."
cd dartsim-master

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Build directory not found. You may need to run configure and make first."
    echo "For Docker, the build should already exist at /headless/dartsim/build"
    exit 1
fi

# Rebuild
cd build
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "DARTSim rebuilt successfully!"
    echo "Restart DARTSim in Docker to use the new version."
else
    echo "Build failed. Check errors above."
    exit 1
fi

