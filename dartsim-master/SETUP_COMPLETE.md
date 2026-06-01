# ✅ DARTSim Setup Complete!

## 🎉 Everything is Ready!

DARTSim has been successfully installed and tested on your Windows machine using Docker.

## 📊 Test Results

The simple-cpp example ran successfully! You can see:
- Simulation executing tactics (DecAlt, GoTight, GoLoose, IncAlt)
- Mission completion with statistics
- Visual representation of the drone route

## 🚀 Quick Start Guide

### Running Examples

From the `dartsim-master` directory:

```powershell
# Run the simple C++ example (as library)
.\run-example.ps1 -Example simple-cpp

# Run the Java example (TCP interface)
.\run-example.ps1 -Example simple-java

# Run the PLA-DART example
.\run-example.ps1 -Example pla-dart
```

Or run directly:

```powershell
docker exec dartsim bash -c "cd /headless/dartsim/examples/simple-cpp && ./run.sh"
```

### Accessing the Visual Interface

1. **Web Browser (Recommended):**
   - Open: http://localhost:6901
   - Password: `vncpassword`
   - This gives you a full desktop environment with terminals

2. **VNC Client:**
   - Connect to: `vnc://localhost:5901`
   - Password: `vncpassword`

### Container Management

```powershell
# Check if container is running
docker ps --filter "name=dartsim"

# Start the container
docker start dartsim

# Stop the container
docker stop dartsim

# Restart the container
docker restart dartsim

# View container logs
docker logs dartsim

# Remove the container (if needed)
docker rm dartsim
```

## 📁 Project Structure

```
dartsim-master/
├── start-dartsim-docker.ps1    # Start DARTSim container
├── run-example.ps1              # Run examples easily
├── fix-docker-virtualization.ps1 # Enable virtualization
├── WINDOWS_SETUP.md             # Detailed Windows setup guide
├── QUICKSTART_WINDOWS.md        # Quick start guide
├── DOCKER_SETUP_INSTRUCTIONS.md # Docker troubleshooting
└── DOCKER_READY.md              # Post-installation guide
```

## 🎮 Example Output Explained

When you run an example, you'll see:

```
current position: X;Y    # Current position of the drone team
executing tactic X       # Adaptation tactic being executed
```

At the end, you'll see:
- A visual ASCII diagram showing the route
- Summary statistics:
  - `out:destroyed=0` - Team was not destroyed
  - `out:targetsDetected=X` - Number of targets detected
  - `out:missionSuccess=0` - Mission success status
  - CSV output with detailed metrics

### ASCII Diagram Symbols

| Symbol | Meaning                 |
|--------|------------------------|
| `#`    | Loose formation        |
| `*`    | Tight formation        |
| `@`    | Loose formation, ECM on |
| `0`    | Tight formation, ECM on |
| `^`    | Threat                 |
| `T`    | Target (not detected)  |
| `X`    | Target (detected)      |

## 🔧 Available Examples

### 1. simple-cpp
- Uses DARTSim as a library
- Runs entirely in one process
- Shows basic adaptation tactics

### 2. simple-java
- Uses TCP interface
- Adaptation manager and simulator in separate processes
- Demonstrates client-server architecture

### 3. pla-dart
- Uses PLA-SDP adaptation approach
- More sophisticated decision-making
- Includes probabilistic model checking

## 📚 Next Steps

1. **Explore the code:**
   ```powershell
   # Open files in the container
   docker exec dartsim bash -c "ls -la /headless/dartsim/examples/simple-cpp/src/"
   ```

2. **Modify examples:**
   - Access container: `docker exec -it dartsim bash`
   - Navigate to examples: `cd /headless/dartsim/examples`
   - Edit and rebuild

3. **Read the documentation:**
   - Original README: `/headless/dartsim/README.md`
   - Source code docs: Various files in `docs/`

4. **Run with different options:**
   ```powershell
   # Custom simulation options
   docker exec dartsim bash -c "cd /headless/dartsim && ./run.sh --num-targets=5 --num-threats=10"
   ```

## 🆘 Troubleshooting

### Container not running?
```powershell
.\start-dartsim-docker.ps1
```

### Can't access web interface?
- Make sure container is running: `docker ps`
- Check if ports are in use: `netstat -an | findstr "6901"`
- Try restarting: `docker restart dartsim`

### Examples not working?
- Check container logs: `docker logs dartsim`
- Verify container is running: `docker ps --filter "name=dartsim"`
- Try rebuilding inside container (if you modified code)

## 📖 Additional Resources

- GitHub Repository: https://github.com/cps-sei/dartsim
- Docker Hub: https://hub.docker.com/r/gabrielmoreno/dartsim
- Paper: Available in `docs/dartsim-paper.pdf`

## ✨ Enjoy!

DARTSim is now ready for you to explore self-adaptation approaches in cyber-physical systems. Have fun experimenting!

