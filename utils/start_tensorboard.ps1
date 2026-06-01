# Start TensorBoard to monitor training progress
# Usage: .\start_tensorboard.ps1 [-LogDir <path>] [-Port <port>]

param(
    [Parameter(Mandatory=$false)]
    [string]$LogDir = "./logs",
    
    [Parameter(Mandatory=$false)]
    [int]$Port = 6006
)

Write-Host "Starting TensorBoard..." -ForegroundColor Green
Write-Host "  Log directory: $LogDir" -ForegroundColor Yellow
Write-Host "  Port: $Port" -ForegroundColor Yellow
Write-Host ""
Write-Host "Access TensorBoard at: http://localhost:$Port" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop TensorBoard" -ForegroundColor Gray
Write-Host ""

# Activate virtual environment and start TensorBoard
& ".\venv\Scripts\Activate.ps1"
& "tensorboard" --logdir $LogDir --port $Port
