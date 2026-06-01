# Docker Desktop Virtualization Setup Instructions

## Current Issue
Docker Desktop is showing "Virtualization support not detected" error.

## Solution Options

### Option 1: Enable Hyper-V (Recommended for this system)

I've enabled Hyper-V for you. You need to:

1. **RESTART YOUR COMPUTER** (required!)
   - Save all your work
   - Restart Windows

2. **After restart, start Docker Desktop**
   - It should now detect virtualization support
   - Wait for Docker Desktop to fully initialize (whale icon in system tray)

3. **Verify Docker is working:**
   ```powershell
   docker --version
   docker ps
   ```

4. **Run DARTSim:**
   ```powershell
   cd dartsim-master
   .\start-dartsim-docker.ps1
   ```

### Option 2: Enable WSL2 Backend (Alternative)

If Hyper-V doesn't work or you prefer WSL2:

1. **Install WSL2:**
   ```powershell
   wsl --install
   ```
   Then restart your computer.

2. **After restart, set WSL2 as default:**
   ```powershell
   wsl --set-default-version 2
   ```

3. **Configure Docker Desktop:**
   - Open Docker Desktop
   - Go to Settings > General
   - Check "Use the WSL 2 based engine"
   - Click "Apply & Restart"

### Option 3: BIOS/UEFI Settings

If virtualization still doesn't work after enabling Hyper-V or WSL2:

1. **Check BIOS/UEFI settings:**
   - Restart and enter BIOS/UEFI (usually F2, F10, F12, or Del during boot)
   - Look for "Virtualization Technology" or "Intel VT-x" / "AMD-V"
   - Enable it if disabled
   - Save and exit

2. **Check if you're in a virtual machine:**
   - If you're running Windows in a VM, nested virtualization must be enabled
   - This depends on your virtualization software (VMware, VirtualBox, Hyper-V, etc.)

## After Docker is Working

Once Docker Desktop starts successfully:

1. **Run the DARTSim setup script:**
   ```powershell
   cd dartsim-master
   .\start-dartsim-docker.ps1
   ```

2. **Access DARTSim:**
   - Open http://localhost:6901 in your web browser
   - Password: `vncpassword`

3. **Run an example:**
   - In the web interface, open a terminal
   - Run: `cd ~/dartsim/examples/simple-cpp && ./run.sh`

## Troubleshooting

### "Virtualization support not detected" persists after restart

1. Check Windows Features:
   - Open "Turn Windows features on or off"
   - Ensure "Hyper-V" is checked
   - Ensure "Windows Subsystem for Linux" is checked
   - Ensure "Virtual Machine Platform" is checked
   - Restart again

2. Check Task Manager:
   - Open Task Manager (Ctrl+Shift+Esc)
   - Go to Performance tab
   - Check CPU section - "Virtualization" should show "Enabled"

3. Check BIOS settings (see Option 3 above)

### Docker Desktop won't start

- Check Windows Event Viewer for errors
- Try running Docker Desktop as Administrator
- Check if antivirus is blocking virtualization features

## Quick Commands Reference

```powershell
# Check virtualization status
systeminfo | Select-String "Hyper-V"

# Check Windows features
Get-WindowsOptionalFeature -Online | Where-Object {$_.FeatureName -like "*Hyper*" -or $_.FeatureName -like "*Virtual*" -or $_.FeatureName -like "*WSL*"}

# Start DARTSim container
cd dartsim-master
.\start-dartsim-docker.ps1

# Stop DARTSim container
docker stop dartsim

# Start stopped container
docker start dartsim
```

