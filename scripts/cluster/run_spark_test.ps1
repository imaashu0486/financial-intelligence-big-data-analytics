# ==============================================================================
# Financial Analytics Platform — Run Spark Job on Standalone Cluster
# ==============================================================================

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Executing Real Distributed Spark Job on Cluster" -ForegroundColor Cyan
Write-Host " Master: spark://hadoop-master:7077 | Workers: hadoop-worker1, hadoop-worker2" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Ensure Input Data Exists in HDFS
Write-Host "`n[1/4] Ensuring input dataset exists in HDFS..." -ForegroundColor Yellow
docker exec hadoop-master hdfs dfs -mkdir -p /financial_analytics/raw /financial_analytics/processed
docker exec hadoop-master hdfs dfs -put -f /opt/project/data/processed/market_features.csv /financial_analytics/raw/market_features.csv

# 2. Clean Previous Spark Output Directory in HDFS
Write-Host "`n[2/4] Cleaning previous HDFS Spark output directory..." -ForegroundColor Yellow
docker exec hadoop-master hdfs dfs -rm -r -f /financial_analytics/processed/spark_cluster_output 2>&1 | Out-Null

# 3. Submit Spark Job to Spark Master
Write-Host "`n[3/4] Submitting Spark job via spark-submit to spark://hadoop-master:7077..." -ForegroundColor Yellow
docker exec hadoop-master /opt/spark/bin/spark-submit \
    --master spark://hadoop-master:7077 \
    --deploy-mode client \
    --driver-memory 512m \
    --executor-memory 512m \
    --executor-cores 1 \
    /opt/project/scripts/cluster/spark_cluster_job.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Spark cluster job failed." -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ Spark cluster job completed successfully!" -ForegroundColor Green

# 4. Verify Parquet Output in HDFS
Write-Host "`n[4/4] Verifying generated Parquet dataset in HDFS..." -ForegroundColor Cyan
docker exec hadoop-master hdfs dfs -ls /financial_analytics/processed/spark_cluster_output

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " Spark Cluster Verification Complete ✨" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

