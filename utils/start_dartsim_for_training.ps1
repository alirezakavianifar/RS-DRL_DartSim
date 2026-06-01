# Start DARTSim for training with process management
# DEPRECATED: This script is for online training mode which has been deprecated.
# Online training is no longer supported. Please use offline RL training instead.
# This ensures DARTSim stays running throughout training (not needed for offline mode)

param(
    [Parameter(Mandatory=$false)]
    [string]$ContainerId = "a65c5bd74a66",
    
    [Parameter(Mandatory=$false)]
    [switch]$StopExisting,
    
    [Parameter(Mandatory=$false)]
    [switch]$Monitor
)

Write-Host "=== DARTSim Process Manager ===" -ForegroundColor Cyan
Write-Host ""

# Stop existing if requested
if ($StopExisting) {
    Write-Host "Stopping existing DARTSim processes..." -ForegroundColor Yellow
    docker exec $ContainerId bash -c "pkill -9 -f dartsim || true" 2>&1 | Out-Null
    Start-Sleep -Seconds 2
}

# Check if already running
$running = docker exec $ContainerId bash -c "ps aux | grep '[d]artsim' || echo ''" 2>&1
if ($running -match "dartsim") {
    Write-Host "[INFO] DARTSim is already running" -ForegroundColor Green
    exit 0
}

Write-Host "Starting DARTSim..." -ForegroundColor Yellow
Write-Host "  Container: $ContainerId" -ForegroundColor Gray
Write-Host "  Port: 5418" -ForegroundColor Gray
Write-Host ""

# Start DARTSim
docker exec -d $ContainerId bash -c "cd /headless/dartsim && nohup ./build/src/dartsim/dartsim > /tmp/dartsim_training.log 2>&1 &"

# Wait for it to start
Write-Host "Waiting for DARTSim to initialize..." -ForegroundColor Yellow
$max_wait = 30
$elapsed = 0
$ready = $false

while ($elapsed -lt $max_wait) {
    Start-Sleep -Seconds 1
    $elapsed++
    
    # Check if process is running
    $running = docker exec $ContainerId bash -c "ps aux | grep '[d]artsim' | grep -v grep || echo ''" 2>&1
    if ($running -match "dartsim") {
        # Check for "Simulator instantiated" message
        $logs = docker exec $ContainerId bash -c "tail -20 /tmp/dartsim_training.log 2>&1 || echo ''" 2>&1
        if ($logs -match "Simulator instantiated") {
            Write-Host "[OK] DARTSim is ready!" -ForegroundColor Green
            $ready = $true
            break
        }
    }
    
    if ($elapsed % 5 -eq 0) {
        Write-Host "  Still waiting... ($elapsed/$max_wait seconds)" -ForegroundColor Gray
    }
}

if (-not $ready) {
    Write-Host "[WARNING] DARTSim may not be ready yet" -ForegroundColor Yellow
    Write-Host "Check logs: docker exec $ContainerId cat /tmp/dartsim_training.log" -ForegroundColor Gray
}

Write-Host ""
Write-Host "DARTSim started for training." -ForegroundColor Green
Write-Host "To monitor: docker exec $ContainerId tail -f /tmp/dartsim_training.log" -ForegroundColor Cyan

if ($Monitor) {
    Write-Host ""
    Write-Host "[WARNING] Monitoring is deprecated - dartsim_manager.py has been removed" -ForegroundColor Yellow
    Write-Host "Online training is no longer supported. Use offline RL training instead." -ForegroundColor Yellow
    # Note: dartsim_manager.py has been removed - monitoring is no longer available
}

