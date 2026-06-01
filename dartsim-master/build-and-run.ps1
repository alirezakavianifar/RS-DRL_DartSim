# PowerShell script to build and run DARTSim in WSL
# This assumes WSL and DARTSim are already set up

param(
    [string]$Example = "simple-cpp",
    [string]$SimOptions = ""
)

Write-Host "DARTSim Build and Run Script" -ForegroundColor Green
Write-Host "===========================" -ForegroundColor Green
Write-Host ""

# Check if WSL is available
try {
    $wslCheck = wsl --list --quiet 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "WSL not found"
    }
} catch {
    Write-Host "ERROR: WSL is not installed or not available!" -ForegroundColor Red
    Write-Host "Please install WSL2 first by running setup-wsl.ps1 as Administrator" -ForegroundColor Yellow
    exit 1
}

# Get the current directory (Windows path)
$currentDir = (Get-Location).Path
$wslPath = $currentDir -replace '^([A-Z]):', '/mnt/$1' -replace '\\', '/'
$wslPath = $wslPath.ToLower()

Write-Host "Current directory (WSL): $wslPath" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the dartsim directory
$inDartsim = Test-Path "src" -PathType Container
if (-not $inDartsim) {
    Write-Host "Warning: src/ directory not found. Make sure you're in the dartsim root directory." -ForegroundColor Yellow
    Write-Host ""
}

# Build if needed
Write-Host "Building DARTSim..." -ForegroundColor Yellow
wsl bash -c "cd '$wslPath' && if [ ! -f build/src/dartsim/dartsim ]; then autoreconf -i && mkdir -p build && cd build && ../configure && make; else echo 'Build already exists, skipping...'; fi"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host ""

# Run the example
Write-Host "Running example: $Example" -ForegroundColor Yellow
Write-Host ""

if ($Example -eq "simple-cpp") {
    $examplePath = "$wslPath/examples/simple-cpp"
    wsl bash -c "cd '$examplePath' && if [ ! -f build/src/simple_cpp ]; then autoreconf -i && mkdir -p build && cd build && ../configure && make; fi && cd '$examplePath' && ./run.sh $SimOptions"
} elseif ($Example -eq "simple-java") {
    $examplePath = "$wslPath/examples/simple-java"
    Write-Host "Starting DARTSim server in background..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "wsl bash -c 'cd $wslPath && ./run.sh'"
    Start-Sleep -Seconds 3
    Write-Host "Starting Java adaptation manager..." -ForegroundColor Cyan
    wsl bash -c "cd '$examplePath' && ./run.sh"
} else {
    Write-Host "Unknown example: $Example" -ForegroundColor Red
    Write-Host "Available examples: simple-cpp, simple-java" -ForegroundColor Yellow
    exit 1
}

