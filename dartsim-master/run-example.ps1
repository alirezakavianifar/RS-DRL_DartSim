# PowerShell script to run DARTSim examples directly from Windows
# This connects to the Docker container and runs examples

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("simple-cpp", "simple-java", "pla-dart")]
    [string]$Example = "simple-cpp"
)

Write-Host "DARTSim Example Runner" -ForegroundColor Green
Write-Host "=====================" -ForegroundColor Green
Write-Host ""

# Check if container is running
$containerRunning = docker ps --filter "name=dartsim" --format "{{.Names}}" | Select-String "dartsim"

if (-not $containerRunning) {
    Write-Host "ERROR: DARTSim container is not running!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Starting container..." -ForegroundColor Yellow
    & "$PSScriptRoot\start-dartsim-docker.ps1"
    Start-Sleep -Seconds 3
}

Write-Host "Running example: $Example" -ForegroundColor Cyan
Write-Host ""

if ($Example -eq "simple-cpp") {
    Write-Host "Running simple-cpp example (DARTSim as library)..." -ForegroundColor Yellow
    docker exec dartsim bash -c "cd /headless/dartsim/examples/simple-cpp && ./run.sh"
} elseif ($Example -eq "simple-java") {
    Write-Host "Running simple-java example (TCP interface)..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "This example requires DARTSim to be running in server mode." -ForegroundColor Cyan
    Write-Host "Starting DARTSim server in background..." -ForegroundColor Yellow
    Start-Job -ScriptBlock { docker exec dartsim bash -c "cd /headless/dartsim && ./run.sh" } | Out-Null
    Start-Sleep -Seconds 2
    Write-Host "Starting Java adaptation manager..." -ForegroundColor Yellow
    docker exec dartsim bash -c "cd /headless/dartsim/examples/simple-java && ./run.sh"
    Write-Host ""
    Write-Host "Stopping DARTSim server..." -ForegroundColor Yellow
    docker exec dartsim pkill -f "dartsim"
} elseif ($Example -eq "pla-dart") {
    Write-Host "Running pla-dart example (PLA-SDP adaptation manager)..." -ForegroundColor Yellow
    docker exec dartsim bash -c "cd /headless/dartsim/examples/pla-dart && ./run.sh"
}

Write-Host ""
Write-Host "Example completed!" -ForegroundColor Green

