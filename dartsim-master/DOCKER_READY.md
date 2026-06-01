# Docker Desktop Setup Complete!

## ✅ What I've Done

1. **Downloaded and installed Docker Desktop**
2. **Enabled Hyper-V virtualization features:**
   - Microsoft-Hyper-V
   - Microsoft-Hyper-V-Online
   - Microsoft-Hyper-V-Offline
   - HypervisorPlatform
3. **Enabled WSL and Virtual Machine Platform**
4. **Created helper scripts for running DARTSim**

## ⚠️ IMPORTANT: RESTART REQUIRED

**You MUST restart your computer** for the virtualization features to take effect!

### After Restart:

1. **Start Docker Desktop** (from Start menu)
   - Wait for it to fully initialize (1-2 minutes)
   - Look for the Docker whale icon in system tray

2. **Verify Docker is working:**
   ```powershell
   docker --version
   ```

3. **Run DARTSim:**
   ```powershell
   cd E:\projects\dart\dartsim-master
   .\start-dartsim-docker.ps1
   ```

## 🚀 Quick Start After Restart

Once Docker Desktop is running:

```powershell
# Navigate to the project
cd E:\projects\dart\dartsim-master

# Start DARTSim in Docker
.\start-dartsim-docker.ps1
```

This will:
- Download the DARTSim Docker image (first time only)
- Start the container
- Show you how to access it

## 🌐 Accessing DARTSim

After running the script, open in your browser:
- **URL:** http://localhost:6901
- **Password:** `vncpassword`

Then in the web terminal:
```bash
cd ~/dartsim/examples/simple-cpp
./run.sh
```

## 📋 If Virtualization Still Doesn't Work

If you still see the virtualization error after restart:

1. **Check BIOS/UEFI Settings:**
   - Restart and press F2/F10/F12/Del during boot
   - Find "Virtualization Technology" or "Intel VT-x" / "AMD-V"
   - Enable it and save

2. **Check Task Manager:**
   - Open Task Manager (Ctrl+Shift+Esc)
   - Performance tab > CPU
   - Look for "Virtualization: Enabled"

3. **Alternative: Use WSL2 Only**
   - If Hyper-V doesn't work, install WSL2 properly:
   ```powershell
   wsl --install
   ```
   - Then configure Docker Desktop to use WSL2 backend only

## 📁 Helper Scripts Created

- `start-dartsim-docker.ps1` - Start DARTSim container
- `fix-docker-virtualization.ps1` - Enable virtualization features
- `DOCKER_SETUP_INSTRUCTIONS.md` - Detailed troubleshooting guide

## Need Help?

See `DOCKER_SETUP_INSTRUCTIONS.md` for detailed troubleshooting steps.

