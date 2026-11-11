# ==============================================================================
# Financial Analytics Platform — Multi-Node Cluster Status Inspector
# ==============================================================================

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Multi-Node Hadoop & Spark Cluster Status Check" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Container Status
Write-Host "`n[1] Docker Container States:" -ForegroundColor Yellow
docker compose ps

# 2. Java Processes inside Master Node
Write-Host "`n[2] Master Node Daemons (hadoop-master):" -ForegroundColor Yellow
docker exec hadoop-master jps -l 2>&1

# 3. Java Processes inside Worker 1
Write-Host "`n[3] Worker 1 Daemons (hadoop-worker1):" -ForegroundColor Yellow
docker exec hadoop-worker1 jps -l 2>&1

# 4. Java Processes inside Worker 2
Write-Host "`n[4] Worker 2 Daemons (hadoop-worker2):" -ForegroundColor Yellow
docker exec hadoop-worker2 jps -l 2>&1

# 5. Summary
Write-Host "`n[5] Web UI Status:" -ForegroundColor Yellow
$endpoints = @(
    @{ Name = "NameNode Web UI"; Url = "http://localhost:9870" },
    @{ Name = "YARN ResourceManager"; Url = "http://localhost:8088" },
    @{ Name = "Spark Master UI"; Url = "http://localhost:8080" },
    @{ Name = "JobHistory Server"; Url = "http://localhost:19888" },
    @{ Name = "Worker 1 DataNode UI"; Url = "http://localhost:9864" },
    @{ Name = "Worker 2 DataNode UI"; Url = "http://localhost:9865" }
)

foreach ($ep in $endpoints) {
    Write-Host ("  {0,-25} : {1}" -f $ep.Name, $ep.Url) -ForegroundColor White
}
Write-Host "======================================================================" -ForegroundColor Cyan

