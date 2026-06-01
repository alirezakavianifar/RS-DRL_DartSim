# PowerShell script to enable virtualization features for Docker Desktop
# Run this script as Administrator

Write-Host "Fixing Docker Desktop Virtualization Support" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "Enabling required Windows features..." -ForegroundColor Yellow
Write-Host ""

# Enable WSL2 if not already enabled
Write-Host "1. Ensuring WSL is enabled..." -ForegroundColor Cyan
$wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
if ($wslFeature.State -ne 'Enabled') {
    Write-Host "   Enabling WSL..." -ForegroundColor Yellow
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart
} else {
    Write-Host "   WSL is already enabled" -ForegroundColor Green
}

# Enable Virtual Machine Platform
Write-Host ""
Write-Host "2. Ensuring Virtual Machine Platform is enabled..." -ForegroundColor Cyan
$vmPlatform = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
if ($vmPlatform.State -ne 'Enabled') {
    Write-Host "   Enabling Virtual Machine Platform..." -ForegroundColor Yellow
    Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart
} else {
    Write-Host "   Virtual Machine Platform is already enabled" -ForegroundColor Green
}

# Optionally enable Hyper-V (alternative to WSL2 backend)
Write-Host ""
Write-Host "3. Checking Hyper-V..." -ForegroundColor Cyan
$hyperV = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -ErrorAction SilentlyContinue
if ($hyperV) {
    if ($hyperV.State -ne 'Enabled') {
        Write-Host "   Hyper-V is disabled. Docker can use WSL2 backend instead (recommended)." -ForegroundColor Yellow
        Write-Host "   If you prefer Hyper-V, uncomment the line below:" -ForegroundColor Cyan
        Write-Host "   # Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -NoRestart" -ForegroundColor Gray
    } else {
        Write-Host "   Hyper-V is enabled" -ForegroundColor Green
    }
} else {
    Write-Host "   Hyper-V feature not available (this is OK, WSL2 is preferred)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Next Steps:" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "1. RESTART YOUR COMPUTER (required for changes to take effect)" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. After restart, ensure WSL2 is set as default:" -ForegroundColor Cyan
Write-Host "   wsl --set-default-version 2" -ForegroundColor White
Write-Host ""
Write-Host "3. Start Docker Desktop and configure it to use WSL2 backend:" -ForegroundColor Cyan
Write-Host "   - Open Docker Desktop" -ForegroundColor White
Write-Host "   - Go to Settings > General" -ForegroundColor White
Write-Host "   - Ensure 'Use the WSL 2 based engine' is checked" -ForegroundColor White
Write-Host "   - Click 'Apply & Restart'" -ForegroundColor White
Write-Host ""
Write-Host "4. If virtualization is still not detected after restart:" -ForegroundColor Cyan
Write-Host "   - Check BIOS/UEFI settings - ensure Virtualization (VT-x/AMD-V) is enabled" -ForegroundColor White
Write-Host "   - Check if you're running in a virtual machine that doesn't support nested virtualization" -ForegroundColor White
Write-Host ""
Write-Host "5. Once Docker is working, run:" -ForegroundColor Cyan
Write-Host "   cd dartsim-master" -ForegroundColor White
Write-Host "   .\start-dartsim-docker.ps1" -ForegroundColor White
Write-Host ""

