# Quick verification script for DARTSim installation
Write-Host "Checking DARTSim Docker setup..." -ForegroundColor Cyan

# Check Docker
Write-Host "`n1. Checking Docker..." -ForegroundColor Yellow
docker --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not available!" -ForegroundColor Red
    exit 1
}

# Check if container exists
Write-Host "`n2. Checking for existing container..." -ForegroundColor Yellow
$exists = docker ps -a --filter "name=dartsim" --format "{{.Names}}"
if ($exists -eq "dartsim") {
    Write-Host "Container exists" -ForegroundColor Green
    $running = docker ps --filter "name=dartsim" --format "{{.Names}}"
    if ($running -eq "dartsim") {
        Write-Host "Container is RUNNING" -ForegroundColor Green
    } else {
        Write-Host "Container exists but is not running" -ForegroundColor Yellow
        Write-Host "Starting container..." -ForegroundColor Cyan
        docker start dartsim
        Start-Sleep -Seconds 3
    }
} else {
    Write-Host "Container does not exist. Creating..." -ForegroundColor Yellow
    Write-Host "This will download the image (first time only, may take a few minutes)..." -ForegroundColor Cyan
    docker run -d -p 5901:5901 -p 6901:6901 --name dartsim gabrielmoreno/dartsim:1.0
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Container created and started!" -ForegroundColor Green
        Start-Sleep -Seconds 3
    } else {
        Write-Host "ERROR: Failed to create container" -ForegroundColor Red
        exit 1
    }
}

# Verify container is running
Write-Host "`n3. Verifying container status..." -ForegroundColor Yellow
$running = docker ps --filter "name=dartsim" --format "{{.Status}}"
if ($running) {
    Write-Host "Container status: $running" -ForegroundColor Green
    Write-Host "`nSUCCESS! DARTSim is running!" -ForegroundColor Green
    Write-Host "`nAccess DARTSim:" -ForegroundColor Cyan
    Write-Host "  Web: http://localhost:6901 (password: vncpassword)" -ForegroundColor White
    Write-Host "  VNC: vnc://localhost:5901 (password: vncpassword)" -ForegroundColor White
} else {
    Write-Host "ERROR: Container is not running!" -ForegroundColor Red
    exit 1
}

