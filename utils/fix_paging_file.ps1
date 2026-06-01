# PowerShell script to check and fix Windows paging file settings
# Must be run as Administrator

Write-Host "Windows Paging File Configuration Tool" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Alternatively, you can manually configure the paging file:" -ForegroundColor Yellow
    Write-Host "1. Press Win+Pause to open System Properties" -ForegroundColor Yellow
    Write-Host "2. Click 'Advanced system settings'" -ForegroundColor Yellow
    Write-Host "3. Click 'Settings' under Performance" -ForegroundColor Yellow
    Write-Host "4. Go to 'Advanced' tab" -ForegroundColor Yellow
    Write-Host "5. Click 'Change' under Virtual memory" -ForegroundColor Yellow
    Write-Host "6. Uncheck 'Automatically manage paging file size'" -ForegroundColor Yellow
    Write-Host "7. Select your system drive (usually C:)" -ForegroundColor Yellow
    Write-Host "8. Select 'Custom size'" -ForegroundColor Yellow
    Write-Host "9. Set Initial size: 8192 MB (8GB)" -ForegroundColor Yellow
    Write-Host "10. Set Maximum size: 16384 MB (16GB)" -ForegroundColor Yellow
    Write-Host "11. Click 'Set' then 'OK'" -ForegroundColor Yellow
    Write-Host "12. Restart your computer" -ForegroundColor Yellow
    exit 1
}

Write-Host "Running as Administrator - OK" -ForegroundColor Green
Write-Host ""

# Get current paging file settings
Write-Host "Current Paging File Configuration:" -ForegroundColor Cyan
$pageFiles = Get-WmiObject -Class Win32_PageFileSetting

if ($pageFiles) {
    foreach ($pf in $pageFiles) {
        Write-Host "  Drive: $($pf.Name)" -ForegroundColor White
        Write-Host "  Initial Size: $($pf.InitialSize) MB" -ForegroundColor White
        Write-Host "  Maximum Size: $($pf.MaximumSize) MB" -ForegroundColor White
        Write-Host "  Auto Managed: $($pf.AutomaticManagedPagefile)" -ForegroundColor White
        Write-Host ""
    }
} else {
    Write-Host "  No paging file found" -ForegroundColor Yellow
}

# Get system memory info
Write-Host "System Memory Information:" -ForegroundColor Cyan
$os = Get-CimInstance Win32_OperatingSystem
$totalRAM = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
$freeRAM = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
$totalVirtual = [math]::Round($os.TotalVirtualMemorySize / 1MB, 2)
$freeVirtual = [math]::Round($os.FreeVirtualMemory / 1MB, 2)

Write-Host "  Total RAM: $totalRAM GB" -ForegroundColor White
Write-Host "  Free RAM: $freeRAM GB" -ForegroundColor White
Write-Host "  Total Virtual Memory: $totalVirtual GB" -ForegroundColor White
Write-Host "  Free Virtual Memory: $freeVirtual GB" -ForegroundColor White
Write-Host ""

# Calculate recommended paging file size
$recommendedInitial = [math]::Max(8192, [math]::Round($totalRAM * 1024 * 1.5))  # 1.5x RAM or 8GB minimum
$recommendedMax = [math]::Max(16384, [math]::Round($totalRAM * 1024 * 3))  # 3x RAM or 16GB minimum

Write-Host "Recommended Paging File Settings:" -ForegroundColor Cyan
Write-Host "  Initial Size: $recommendedInitial MB ($([math]::Round($recommendedInitial/1024, 1)) GB)" -ForegroundColor Green
Write-Host "  Maximum Size: $recommendedMax MB ($([math]::Round($recommendedMax/1024, 1)) GB)" -ForegroundColor Green
Write-Host ""

# Check if current settings are sufficient
$needsFix = $false
if ($pageFiles) {
    foreach ($pf in $pageFiles) {
        if ($pf.MaximumSize -lt $recommendedMax) {
            $needsFix = $true
            Write-Host "Current paging file size ($($pf.MaximumSize) MB) is less than recommended ($recommendedMax MB)" -ForegroundColor Yellow
        }
    }
} else {
    $needsFix = $true
    Write-Host "No paging file configured or auto-managed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Note: To modify paging file settings, you need to use System Properties:" -ForegroundColor Yellow
Write-Host "This script can only check current settings (modification requires registry changes)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Manual Steps to Fix:" -ForegroundColor Cyan
Write-Host "1. Press Win+Pause → Advanced system settings → Settings (Performance)" -ForegroundColor White
Write-Host "2. Advanced tab → Change (Virtual memory)" -ForegroundColor White
Write-Host "3. Uncheck 'Automatically manage paging file size'" -ForegroundColor White
Write-Host "4. Select system drive → Custom size" -ForegroundColor White
Write-Host "5. Set Initial: $recommendedInitial MB, Maximum: $recommendedMax MB" -ForegroundColor White
Write-Host "6. Click Set → OK → Restart computer" -ForegroundColor White
Write-Host ""

# Alternative: Try to set via registry (requires admin)
Write-Host "Attempting to configure via registry..." -ForegroundColor Cyan
$systemDrive = $env:SystemDrive

try {
    # Get current registry value
    $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
    $autoManage = Get-ItemProperty -Path $regPath -Name "PagingFiles" -ErrorAction SilentlyContinue
    
    Write-Host "Registry path: $regPath" -ForegroundColor White
    
    # Note: Actually modifying paging file requires more complex registry changes
    # and may require system restart. It's safer to guide user to do it manually.
    Write-Host ""
    Write-Host "For safety, please use the manual method above." -ForegroundColor Yellow
    Write-Host "Registry modification requires careful handling and system restart." -ForegroundColor Yellow
    
} catch {
    Write-Host "Could not access registry (may need different permissions)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Script complete. Please follow manual steps above if paging file needs adjustment." -ForegroundColor Cyan



