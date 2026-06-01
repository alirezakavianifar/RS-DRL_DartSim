<#
.SYNOPSIS
    Start the gabrielmoreno/dartsim:1.0 Docker container and launch DARTSim
    in TCP-server mode, exposing port 5418 for the Python RL client.

.PARAMETER ContainerName
    Logical name given to the Docker container.  Default: dartsim

.PARAMETER SimArgs
    Extra arguments forwarded to DARTSim run.sh
    e.g. "--seed=42 --map-size=40"

.PARAMETER Timeout
    Seconds to wait for DARTSim to become ready.  Default: 30

.EXAMPLE
    .\utils\start_dartsim_live.ps1
    .\utils\start_dartsim_live.ps1 -SimArgs "--seed=42 --map-size=40"
#>

param(
    [string]$ContainerName = "dartsim",
    [string]$SimArgs       = "",
    [int]   $Timeout       = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step([string]$msg) { Write-Host "  $msg" -ForegroundColor Cyan }
function Write-OK([string]$msg)   { Write-Host "  [OK]  $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "  [!!]  $msg" -ForegroundColor Yellow }
function Write-Fail([string]$msg) { Write-Host "  [FAIL] $msg" -ForegroundColor Red }

# --------------------------------------------------------------------
# 1. Check Docker daemon
# --------------------------------------------------------------------

Write-Host ""
Write-Host "DARTSim Live Environment Setup" -ForegroundColor White
Write-Host "================================" -ForegroundColor White
Write-Host ""

Write-Step "Checking Docker daemon..."
try {
    $null = docker info 2>&1
    Write-OK "Docker daemon is running"
} catch {
    Write-Fail "Docker daemon is not running. Please start Docker Desktop."
    exit 1
}

# --------------------------------------------------------------------
# 2. Container lifecycle
# --------------------------------------------------------------------

Write-Step "Checking container '$ContainerName'..."

# Temporarily relax error preference so a missing container doesn't throw
$_prev = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$status = (docker inspect --format '{{.State.Status}}' $ContainerName 2>&1) -join ''
$exists = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $_prev

if (-not $exists) {
    Write-Step "Pulling gabrielmoreno/dartsim:1.0 (first run -- may take a few minutes)..."
    docker pull gabrielmoreno/dartsim:1.0
    if ($LASTEXITCODE -ne 0) { Write-Fail "docker pull failed"; exit 1 }

    Write-Step "Creating container '$ContainerName' (ports 5418, 5901, 6901)..."
    docker run -d `
        -p 5418:5418 `
        -p 5901:5901 `
        -p 6901:6901 `
        --name $ContainerName `
        gabrielmoreno/dartsim:1.0
    if ($LASTEXITCODE -ne 0) { Write-Fail "docker run failed"; exit 1 }
    Write-OK "Container created and started"
} elseif ($status -eq "running") {
    Write-OK "Container is already running"
} else {
    Write-Step "Starting stopped container (state: $status)..."
    docker start $ContainerName | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Fail "docker start failed"; exit 1 }
    Write-OK "Container started"
}

# --------------------------------------------------------------------
# 3. Kill any leftover DARTSim process
# --------------------------------------------------------------------

Write-Step "Stopping any existing DARTSim process in the container..."
docker exec $ContainerName bash -c "pkill -f 'dartsim' 2>/dev/null ; sleep 0.3; true" | Out-Null
Write-OK "Cleared"

# --------------------------------------------------------------------
# 4. Start DARTSim in TCP-server mode
# --------------------------------------------------------------------

$runCmd = "cd /headless/dartsim && nohup ./build/src/dartsim/dartsim $SimArgs > /tmp/dartsim.log 2>&1 &"
Write-Step "Launching DARTSim..."
docker exec -d $ContainerName bash -c $runCmd
if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to start DARTSim inside container"; exit 1 }

# --------------------------------------------------------------------
# 5. Wait for DARTSim process to be running (do NOT do a TCP connect here
#    because DARTSim accepts exactly ONE client per run -- a probe would
#    consume the session before the Python client can use it)
# --------------------------------------------------------------------

Write-Step "Waiting for DARTSim process to start..."

$ready   = $false
$elapsed = 0

while ($elapsed -lt $Timeout) {
    Start-Sleep -Milliseconds 500
    $elapsed++

    # Check if the dartsim binary is listed in the container's process table
    $procs = docker exec $ContainerName bash -c "pgrep -x dartsim 2>/dev/null || echo ''" 2>$null
    if ($procs -match '\d') {
        $ready = $true
        break
    }

    if ($elapsed % 10 -eq 0) {
        Write-Step "  still waiting... ($elapsed s)"
        $log = docker exec $ContainerName bash -c "tail -3 /tmp/dartsim.log 2>/dev/null || echo '(no log yet)'" 2>$null
        Write-Host "    $log" -ForegroundColor DarkGray
    }
}

if (-not $ready) {
    Write-Fail "DARTSim process did not start after $Timeout s."
    Write-Host ""
    Write-Host "Container log:" -ForegroundColor Yellow
    docker exec $ContainerName bash -c "cat /tmp/dartsim.log 2>/dev/null || echo '(empty)'"
    exit 1
}

# --------------------------------------------------------------------
# 6. Done
# --------------------------------------------------------------------

Write-Host ""
Write-OK "DARTSim process is running (port 5418 ready for one client per episode)"
Write-Host ""
Write-Host "  VNC viewer :  vnc://localhost:5901  (password: vncpassword)" -ForegroundColor DarkGray
Write-Host "  HTTP viewer:  http://localhost:6901  (password: vncpassword)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "To train the RS-DRL agent online:" -ForegroundColor White
Write-Host "  python scripts\train_online.py --timesteps 50000" -ForegroundColor White
Write-Host ""
Write-Host "To collect fresh offline data with all 8 actions:" -ForegroundColor White
Write-Host "  python scripts\collect_offline_data.py --num-episodes 1000 --use-tcp" -ForegroundColor White
Write-Host ""
