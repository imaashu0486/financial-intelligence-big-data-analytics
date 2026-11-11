# ==============================================================================
# Financial Analytics Platform — Spark Cluster Verification (Test D)
# ==============================================================================

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " TEST D: Apache Spark Standalone Cluster Verification" -ForegroundColor Cyan
Write-Host " Requirement: 2 Spark Workers Registered with Master" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Query Spark Master Web UI / Alive Workers
Write-Host "`n[1/3] Checking Spark Master daemon and registered workers..." -ForegroundColor Yellow
$sparkMasterLog = docker exec hadoop-master cat /opt/spark/logs/*master*.out 2>&1
$workerCount = 0

# Count worker registration messages in master log
$workerRegistrations = ($sparkMasterLog | Select-String "Registering worker hadoop-worker")
if ($workerRegistrations) {
    $workerCount = $workerRegistrations.Count
}

Write-Host "  Registered Spark Workers detected in log: $workerCount" -ForegroundColor White
if ($workerCount -ge 2) {
    Write-Host "  ✅ SUCCESS: Both Spark Workers registered with Spark Master!" -ForegroundColor Green
} else {
    Write-Host "  Checking active worker processes..." -ForegroundColor Gray
    docker exec hadoop-worker1 jps | Select-String "Worker" | Out-String | Write-Host
    docker exec hadoop-worker2 jps | Select-String "Worker" | Out-String | Write-Host
}

# 2. Test Spark Connectivity via spark-submit in master container
Write-Host "`n[2/3] Testing Spark Master connectivity via spark-submit..." -ForegroundColor Yellow
docker exec hadoop-master /opt/spark/bin/spark-submit \
    --master spark://hadoop-master:7077 \
    --class org.apache.spark.examples.SparkPi \
    /opt/spark/examples/jars/spark-examples_*.jar 10 2>&1 | Select-String "Pi is roughly" | Out-String | Write-Host

# 3. Web UI Reference
Write-Host "`n[3/3] Spark Cluster Web Endpoints:" -ForegroundColor Cyan
Write-Host "  Spark Master Web UI:  http://localhost:8080" -ForegroundColor White
Write-Host "  Worker 1 Web UI:      http://localhost:8081" -ForegroundColor White
Write-Host "  Worker 2 Web UI:      http://localhost:8082" -ForegroundColor White

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " Spark Cluster Verification Complete ✨" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

