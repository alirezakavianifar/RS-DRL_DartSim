# Quick Start Guide - Running DARTSim on Windows

## Prerequisites

DARTSim requires a Linux-like environment. You need either:
- **WSL2** (Windows Subsystem for Linux) - Recommended, free
- **Docker Desktop** - Alternative option

## Quick Setup with WSL2

### Step 1: Install WSL2

Open **PowerShell as Administrator** and run:

```powershell
.\setup-wsl.ps1
```

Or manually:

```powershell
wsl --install
```

**Restart your computer** after installation.

### Step 2: Launch Ubuntu

1. After restart, open "Ubuntu" from the Start menu
2. Create a username and password
3. Update the system (optional but recommended):

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

### Step 3: Set Up DARTSim

In the Ubuntu terminal, navigate to the project and run the setup script:

```bash
# If you're in Windows, access the project via /mnt
cd /mnt/e/projects/dart/dartsim-master

# Or copy it to your home directory
cp -r /mnt/e/projects/dart/dartsim-master ~/dartsim
cd ~/dartsim

# Run the setup script
bash setup-dartsim.sh
```

This will:
- Install all required dependencies
- Build DARTSim from source
- Set up the examples

### Step 4: Run an Example

```bash
# Simple C++ example
cd ~/dartsim/examples/simple-cpp
../run.sh
```

Or from Windows PowerShell (in the dartsim-master directory):

```powershell
.\build-and-run.ps1 -Example simple-cpp
```

## Alternative: Using Docker Desktop

If you prefer Docker:

1. Install Docker Desktop from https://www.docker.com/products/docker-desktop
2. Run the container:

```powershell
docker run -d -p 5901:5901 -p 6901:6901 --name dartsim gabrielmoreno/dartsim:1.0
```

3. Access via web browser: http://localhost:6901 (password: `vncpassword`)

## What to Expect

When you run the simple-cpp example, you should see output like:

```
current position: 0;0
executing tactic DecAlt
current position: 1;0
executing tactic GoTight
...

Total targets detected: 1
### * *###### # # ## * # # #* *         
   * #       # # *  # # # *  # #        
                                #      #
                                 ###### 
   ^  ^          ^        ^ ^ ^         
              T           T      X    T 
```

## Troubleshooting

**WSL installation fails?**
- Enable "Windows Subsystem for Linux" in "Turn Windows features on or off"
- Ensure Virtual Machine Platform is enabled
- Restart and try again

**Build errors?**
- Make sure all dependencies are installed: `sudo apt-get install -y libboost-all-dev libyaml-cpp-dev make automake autoconf g++ default-jdk ant wget libtool`
- Try cleaning: `rm -rf build && mkdir build && cd build && ../configure && make`

**Can't find the project?**
- From Ubuntu: `cd /mnt/e/projects/dart/dartsim-master`
- Or copy to home: `cp -r /mnt/e/projects/dart/dartsim-master ~/dartsim`

For more details, see `WINDOWS_SETUP.md`.

