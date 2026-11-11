# ==============================================================================
# Financial Analytics Platform — YARN Multi-Node Verification (Test C)
# ==============================================================================

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " TEST C: YARN Distributed Resource Manager Verification" -ForegroundColor Cyan
Write-Host " Requirement: 2 Active NodeManagers Registered" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Query Registered YARN Nodes
Write-Host "`n[1/3] Querying active YARN nodes (yarn node -list -all)..." -ForegroundColor Yellow
$nodeList = docker exec hadoop-master yarn node -list -all 2>&1
$nodeList | Out-String | Write-Host -ForegroundColor Gray

# Extract Total Nodes
if ($nodeList -match "Total Nodes:(\d+)") {
    $activeCount = [int]$matches[1]
    Write-Host "  Active YARN NodeManagers: $activeCount" -ForegroundColor White
    if ($activeCount -ge 2) {
        Write-Host "  ✅ SUCCESS: Both NodeManagers registered with ResourceManager!" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ WARNING: Expected 2 active NodeManagers, found $activeCount." -ForegroundColor Red
    }
} else {
    Write-Host "  ⚠️ Could not parse YARN active node count directly." -ForegroundColor Yellow
}

# 2. Query YARN Cluster State & Capacity
Write-Host "`n[2/3] Checking cluster memory and vcore capacity..." -ForegroundColor Yellow
docker exec hadoop-master yarn top -n 1 2>&1 | Select-Object -First 10 | Out-String | Write-Host -ForegroundColor Gray

# 3. YARN Web UI Reference
Write-Host "`n[3/3] YARN ResourceManager Web Interface:" -ForegroundColor Cyan
Write-Host "  Open in Browser: http://localhost:8088" -ForegroundColor White
Write-Host "  View active nodes: http://localhost:8088/cluster/nodes" -ForegroundColor White

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " YARN Cluster Verification Complete ✨" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

