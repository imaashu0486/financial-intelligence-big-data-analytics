# ==============================================================================
# Financial Analytics Platform — Run Full Financial Analytics Pipeline on Cluster
# ==============================================================================

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Running Full PySpark Financial Analytics Pipeline on Cluster" -ForegroundColor Cyan
Write-Host " Master Node: hadoop-master" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Ensure Local Processed Data Exists
$localMarketData = "$ProjectRoot\data\processed\market_features.csv"
if (-not (Test-Path $localMarketData)) {
    Write-Host "Local market_features.csv not found. Running Phase 1 data ingestion first..." -ForegroundColor Yellow
    python scripts/ingest_data.py --source sample
}

# 2. Upload Input Data to HDFS
Write-Host "`n[1/3] Uploading market data to HDFS /financial_analytics/raw/..." -ForegroundColor Yellow
docker exec hadoop-master hdfs dfs -mkdir -p /financial_analytics/raw /financial_analytics/processed/spark
docker exec hadoop-master hdfs dfs -put -f /opt/project/data/processed/market_features.csv /financial_analytics/raw/market_features.csv

# 3. Execute Pipeline Inside Master Container
Write-Host "`n[2/3] Executing SparkPipeline on hadoop-master..." -ForegroundColor Yellow
docker exec hadoop-master python3 /opt/project/scripts/run_spark.py --input /opt/project/data/processed/market_features.csv

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Financial analytics pipeline failed on cluster." -ForegroundColor Red
    exit 1
}

Write-Host "`n[3/3] Output Parquet files generated in cluster storage:" -ForegroundColor Cyan
docker exec hadoop-master ls -la /opt/project/data/processed/spark

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " Full Financial Analytics Pipeline Completed on Cluster ✨" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

