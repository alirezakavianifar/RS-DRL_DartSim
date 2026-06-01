# PowerShell script to start DARTSim in Docker
# Make sure Docker Desktop is running before executing this script

Write-Host "DARTSim Docker Setup" -ForegroundColor Green
Write-Host "===================" -ForegroundColor Green
Write-Host ""

# Check if Docker is available
$dockerAvailable = $false
try {
    $null = docker --version 2>$null
    $dockerAvailable = $true
} catch {
    $dockerAvailable = $false
}

if (-not $dockerAvailable) {
    Write-Host "ERROR: Docker is not available!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please:" -ForegroundColor Yellow
    Write-Host "1. Make sure Docker Desktop is installed" -ForegroundColor Yellow
    Write-Host "2. Start Docker Desktop from the Start menu" -ForegroundColor Yellow
    Write-Host "3. Wait for Docker Desktop to fully initialize (check system tray icon)" -ForegroundColor Yellow
    Write-Host "4. Run this script again" -ForegroundColor Yellow
    exit 1
}

Write-Host "Docker is available!" -ForegroundColor Green
docker --version
Write-Host ""

# Check if container already exists
$containerExists = docker ps -a --filter "name=dartsim" --format "{{.Names}}" | Select-String "dartsim"

if ($containerExists) {
    Write-Host "DARTSim container already exists." -ForegroundColor Cyan
    
    # Check if it's running
    $containerRunning = docker ps --filter "name=dartsim" --format "{{.Names}}" | Select-String "dartsim"
    
    if ($containerRunning) {
        Write-Host "Container is already running!" -ForegroundColor Green
    } else {
        Write-Host "Starting existing container..." -ForegroundColor Yellow
        docker start dartsim
        Start-Sleep -Seconds 2
    }
} else {
    Write-Host "Creating and starting DARTSim container..." -ForegroundColor Yellow
    Write-Host "This will download the DARTSim image from Docker Hub (first time only)..." -ForegroundColor Cyan
    Write-Host ""
    
    docker run -d -p 5901:5901 -p 6901:6901 --name dartsim gabrielmoreno/dartsim:1.0
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Container started successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "ERROR: Failed to start container!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "DARTSim is now running!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access DARTSim in one of these ways:" -ForegroundColor Cyan
Write-Host ""
Write-Host "Option 1: Web Browser (Recommended)" -ForegroundColor Yellow
Write-Host "  Open: http://localhost:6901" -ForegroundColor White
Write-Host "  Password: vncpassword" -ForegroundColor White
Write-Host ""
Write-Host "Option 2: VNC Client" -ForegroundColor Yellow
Write-Host "  Connect to: vnc://localhost:5901" -ForegroundColor White
Write-Host "  Password: vncpassword" -ForegroundColor White
Write-Host ""
Write-Host "Once connected, open a terminal and run:" -ForegroundColor Cyan
Write-Host "  cd ~/dartsim/examples/simple-cpp" -ForegroundColor White
Write-Host "  ./run.sh" -ForegroundColor White
Write-Host ""
Write-Host "To stop the container:" -ForegroundColor Cyan
Write-Host "  docker stop dartsim" -ForegroundColor White
Write-Host ""
Write-Host "To remove the container:" -ForegroundColor Cyan
Write-Host "  docker rm dartsim" -ForegroundColor White
Write-Host ""

