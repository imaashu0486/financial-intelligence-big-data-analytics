# ==============================================================================
# Financial Analytics Platform — HDFS Multi-Node Verification (Test B)
# ==============================================================================

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " TEST B: HDFS Distributed Cluster Verification" -ForegroundColor Cyan
Write-Host " Requirement: 2 Live DataNodes & Replication Factor 2" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. HDFS dfsadmin Report
Write-Host "`n[1/5] Querying HDFS cluster report (hdfs dfsadmin -report)..." -ForegroundColor Yellow
$report = docker exec hadoop-master hdfs dfsadmin -report 2>&1
$report | Out-String | Write-Host -ForegroundColor Gray

# Extract Live DataNodes count
$liveNodesMatch = ($report | Select-String "Live datanodes \(([0-9]+)\)")
if ($liveNodesMatch -match "Live datanodes \((\d+)\)") {
    $liveCount = [int]$matches[1]
    Write-Host "  Live DataNodes Count: $liveCount" -ForegroundColor White
    if ($liveCount -ge 2) {
        Write-Host "  ✅ SUCCESS: Both DataNodes registered and operational!" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ WARNING: Expected 2 live DataNodes, found $liveCount." -ForegroundColor Red
    }
} else {
    Write-Host "  ⚠️ Could not parse Live DataNodes count directly. Check report above." -ForegroundColor Yellow
}

# 2. Create HDFS Directory
Write-Host "`n[2/5] Creating HDFS test directory (/financial_analytics/raw)..." -ForegroundColor Yellow
docker exec hadoop-master hdfs dfs -mkdir -p /financial_analytics/raw /financial_analytics/processed
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ HDFS directories created successfully." -ForegroundColor Green
}

# 3. Upload Project Market Data to HDFS with Replication Factor 2
Write-Host "`n[3/5] Uploading market features to HDFS (/opt/project/data/processed/market_features.csv)..." -ForegroundColor Yellow
docker exec hadoop-master hdfs dfs -put -f /opt/project/data/processed/market_features.csv /financial_analytics/raw/market_features.csv

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ File uploaded to HDFS: /financial_analytics/raw/market_features.csv" -ForegroundColor Green
} else {
    Write-Host "  ❌ Failed to upload market_features.csv to HDFS." -ForegroundColor Red
}

# 4. Enforce & Verify Replication Factor 2
Write-Host "`n[4/5] Setting & verifying replication factor = 2..." -ForegroundColor Yellow
docker exec hadoop-master hdfs dfs -setrep -w 2 /financial_analytics/raw/market_features.csv
docker exec hadoop-master hdfs fsck /financial_analytics/raw/market_features.csv -files -blocks -locations

# 5. Read Sample Data Back from HDFS
Write-Host "`n[5/5] Reading first 5 rows back from HDFS..." -ForegroundColor Yellow
docker exec hadoop-master hdfs dfs -cat /financial_analytics/raw/market_features.csv | Select-Object -First 6

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " HDFS Multi-Node Verification Complete ✨" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

