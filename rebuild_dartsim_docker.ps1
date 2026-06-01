# Rebuild DARTSim in Docker container with improved connection handling

Write-Host "Rebuilding DARTSim in Docker container..." -ForegroundColor Cyan

$containerId = "a65c5bd74a66"
$dartsimPath = "/headless/dartsim"

# Check if container is running
$containerRunning = docker ps --filter "id=$containerId" --format "{{.ID}}"
if (-not $containerRunning) {
    Write-Host "ERROR: Container $containerId is not running!" -ForegroundColor Red
    exit 1
}

Write-Host "Container found. Rebuilding DARTSim..." -ForegroundColor Green

# Rebuild DARTSim
docker exec $containerId bash -c "cd $dartsimPath/build && make -j`$(nproc)"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDARTSim rebuilt successfully!" -ForegroundColor Green
    Write-Host "Restart DARTSim to use the new version." -ForegroundColor Yellow
    Write-Host "You can restart it using dartsim_manager or manually." -ForegroundColor Yellow
} else {
    Write-Host "`nBuild failed. Check errors above." -ForegroundColor Red
    exit 1
}

