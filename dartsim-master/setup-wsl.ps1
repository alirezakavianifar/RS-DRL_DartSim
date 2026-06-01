# PowerShell script to set up WSL2 for DARTSim
# Run this script as Administrator

Write-Host "DARTSim WSL2 Setup Script" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Check if WSL is already installed
$wslInstalled = $false
try {
    $wslVersion = wsl --version 2>$null
    $wslInstalled = $true
} catch {
    $wslInstalled = $false
}

if (-not $wslInstalled) {
    Write-Host "Installing WSL2..." -ForegroundColor Yellow
    wsl --install
    Write-Host ""
    Write-Host "WSL2 installation initiated. Please:" -ForegroundColor Green
    Write-Host "1. Restart your computer" -ForegroundColor Yellow
    Write-Host "2. Launch 'Ubuntu' from Start menu and complete setup" -ForegroundColor Yellow
    Write-Host "3. Run the setup-dartsim.sh script inside Ubuntu" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After restart, you can also run: wsl bash setup-dartsim.sh" -ForegroundColor Cyan
} else {
    Write-Host "WSL2 is already installed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To set up DARTSim in WSL, run:" -ForegroundColor Cyan
    Write-Host "  wsl bash setup-dartsim.sh" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Current WSL distributions:" -ForegroundColor Cyan
wsl --list --verbose

