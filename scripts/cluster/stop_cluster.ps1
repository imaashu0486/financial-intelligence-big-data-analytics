# ==============================================================================
# Financial Analytics Platform — Stop Distributed Hadoop & Spark Cluster
# ==============================================================================

[CmdletBinding()]
param(
    [switch]$Volumes = $false
)

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Stopping Multi-Node Hadoop & Spark Docker Cluster" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

if ($Volumes) {
    Write-Host "Stopping containers and removing persistent volumes (Data Reset)..." -ForegroundColor Yellow
    docker compose down -v
} else {
    Write-Host "Stopping containers (preserving HDFS & MongoDB volume data)..." -ForegroundColor Yellow
    docker compose down
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Cluster stopped successfully." -ForegroundColor Green
} else {
    Write-Host "❌ Failed to cleanly stop containers." -ForegroundColor Red
}

