# ==============================================================================
# Financial Analytics Platform — Start Distributed Hadoop & Spark Cluster
# ==============================================================================

[CmdletBinding()]
param(
    [switch]$Build = $false
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Starting Multi-Node Hadoop & Spark Docker Cluster" -ForegroundColor Cyan
Write-Host " Host: Windows (Single-Host 3-Container Topology)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Verify Docker Engine Availability
Write-Host "`n[1/4] Checking Docker daemon status..." -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker engine not reachable."
    }
    Write-Host "  ✅ Docker daemon is running." -ForegroundColor Green
} catch {
    Write-Host "  ❌ Docker Desktop is not currently running or the service is stopped." -ForegroundColor Red
    Write-Host "`n  Instructions to start Docker Desktop on Windows:" -ForegroundColor Yellow
    Write-Host "    1. Open Start Menu -> Search 'Docker Desktop' -> Click to Launch." -ForegroundColor White
    Write-Host "    2. Wait for Docker icon in taskbar tray to turn steady (green/running)." -ForegroundColor White
    Write-Host "    3. Re-run this script: .\scripts\cluster\start_cluster.ps1" -ForegroundColor White
    exit 1
}

# 2. Launch Docker Compose Cluster
Write-Host "`n[2/4] Starting Docker containers (hadoop-master, hadoop-worker1, hadoop-worker2, mongodb)..." -ForegroundColor Yellow
if ($Build) {
    docker compose up -d --build
} else {
    docker compose up -d
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Failed to launch containers." -ForegroundColor Red
    exit 1
}

# 3. Wait for Services to Initialize
Write-Host "`n[3/4] Waiting for Hadoop & Spark Master services to initialize..." -ForegroundColor Yellow
$timeoutSeconds = 60
$elapsed = 0
$masterReady = $false

while ($elapsed -lt $timeoutSeconds) {
    Start-Sleep -Seconds 3
    $elapsed += 3
    
    # Check NameNode Web UI port 9870
    $res = docker exec hadoop-master nc -z localhost 9870 2>&1
    if ($LASTEXITCODE -eq 0) {
        $masterReady = $true
        break
    }
    Write-Host "  Waiting for NameNode... ($elapsed s)" -ForegroundColor Gray
}

if (-not $masterReady) {
    Write-Host "  ⚠️ NameNode took longer than expected. Check logs using: docker logs hadoop-master" -ForegroundColor Yellow
} else {
    Write-Host "  ✅ NameNode and Master daemons are online!" -ForegroundColor Green
}

# 4. Display Cluster Web Endpoints
Write-Host "`n[4/4] Cluster Services & Web Endpoints:" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  HDFS NameNode UI:       http://localhost:9870" -ForegroundColor White
Write-Host "  YARN ResourceManager:   http://localhost:8088" -ForegroundColor White
Write-Host "  Spark Master Web UI:    http://localhost:8080" -ForegroundColor White
Write-Host "  MapReduce JobHistory:   http://localhost:19888" -ForegroundColor White
Write-Host "  Worker 1 DataNode UI:   http://localhost:9864" -ForegroundColor White
Write-Host "  Worker 2 DataNode UI:   http://localhost:9865" -ForegroundColor White
Write-Host "  Worker 1 NodeManager:   http://localhost:8042" -ForegroundColor White
Write-Host "  Worker 2 NodeManager:   http://localhost:8043" -ForegroundColor White
Write-Host "  Worker 1 Spark UI:      http://localhost:8081" -ForegroundColor White
Write-Host "  Worker 2 Spark UI:      http://localhost:8082" -ForegroundColor White
Write-Host "  MongoDB Port:           localhost:27017" -ForegroundColor White
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray

Write-Host "`nNext Step: Run verification scripts:" -ForegroundColor Yellow
Write-Host "  .\scripts\cluster\cluster_status.ps1" -ForegroundColor Gray
Write-Host "  .\scripts\cluster\verify_hdfs.ps1" -ForegroundColor Gray
Write-Host "  .\scripts\cluster\verify_yarn.ps1" -ForegroundColor Gray
Write-Host "  .\scripts\cluster\verify_spark.ps1" -ForegroundColor Gray

