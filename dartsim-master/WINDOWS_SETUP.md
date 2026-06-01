# DARTSim Windows Setup Guide

DARTSim is designed for Linux/Unix environments. On Windows, you have three options:

## Option 1: Using WSL2 (Recommended - Free and Easy)

Windows Subsystem for Linux (WSL2) allows you to run a Linux environment directly on Windows.

### Step 1: Install WSL2

Open PowerShell as Administrator and run:

```powershell
wsl --install
```

This will install WSL2 with Ubuntu by default. After installation, restart your computer.

### Step 2: Set Up Ubuntu

1. After restart, launch "Ubuntu" from the Start menu
2. Create a username and password when prompted
3. Update the system:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 3: Install Required Dependencies

In your Ubuntu terminal:

```bash
sudo apt-get install -y libboost-all-dev libyaml-cpp-dev make automake autoconf g++ default-jdk ant wget libtool
```

### Step 4: Copy DARTSim to WSL

From Windows PowerShell (in the project directory):

```powershell
# Note the Windows path format for WSL
wsl cp -r /mnt/e/projects/dart/dartsim-master ~/dartsim
```

Or from Ubuntu terminal:

```bash
cp -r /mnt/e/projects/dart/dartsim-master ~/dartsim
cd ~/dartsim
```

### Step 5: Build DARTSim

In Ubuntu terminal:

```bash
cd ~/dartsim
autoreconf -i
mkdir -p build
cd build
../configure
make
```

### Step 6: Run Examples

```bash
# Simple C++ example
cd ~/dartsim/examples/simple-cpp
../run.sh

# Or build and run manually:
cd ~/dartsim/examples/simple-cpp
autoreconf -i
mkdir -p build
cd build
../configure
make
cd ..
./run.sh
```

## Option 2: Using Docker Desktop (Alternative)

### Step 1: Install Docker Desktop

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop
2. Install and restart your computer
3. Launch Docker Desktop

### Step 2: Run DARTSim Container

Open PowerShell or Command Prompt:

```powershell
docker run -d -p 5901:5901 -p 6901:6901 --name dartsim gabrielmoreno/dartsim:1.0
```

### Step 3: Access the Container

You can access DARTSim in two ways:

**Via Web Browser (Recommended):**
- Open: http://localhost:6901
- Password: `vncpassword`

**Via VNC Client:**
- Connect to: `vnc://localhost:5901`
- Password: `vncpassword`

Once inside, open a terminal and run:

```bash
cd ~/dartsim/examples/simple-cpp
./run.sh
```

## Option 3: Native Windows Build (Advanced)

This requires MSYS2 and manual library installation. Not recommended unless you have specific needs.

### Prerequisites

1. Install MSYS2 from: https://www.msys2.org/
2. Install Visual Studio Build Tools or MinGW-w64

### Setup MSYS2

In MSYS2 terminal:

```bash
pacman -Syu
pacman -S base-devel gcc make automake autoconf libtool
pacman -S mingw-w64-x86_64-boost mingw-w64-x86_64-yaml-cpp
```

Then navigate to the project and build (may require modifications to build files).

## Quick Start Scripts

See `setup-wsl.ps1` and `build-and-run.ps1` for automated setup and execution scripts.

## Troubleshooting

### WSL Issues

- If `wsl --install` fails, you may need to enable Windows features manually:
  - Open "Turn Windows features on or off"
  - Enable "Windows Subsystem for Linux" and "Virtual Machine Platform"
  - Restart and try again

### Build Errors

- Ensure all dependencies are installed
- Try cleaning the build directory: `rm -rf build && mkdir build`

### Permission Errors

- Use `sudo` for system-wide installations
- Check file permissions: `chmod +x run.sh`

