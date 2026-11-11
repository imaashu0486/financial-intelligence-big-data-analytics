# ==============================================================================
# Financial Analytics Platform — Run MapReduce on YARN Cluster
# ==============================================================================

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Executing Real Hadoop Streaming MapReduce Job on YARN" -ForegroundColor Cyan
Write-Host " Cluster: hadoop-master (RM) + hadoop-worker1/2 (NM)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Upload Cleaned News to HDFS
Write-Host "`n[1/4] Ensuring input dataset exists in HDFS (/financial_analytics/raw)..." -ForegroundColor Yellow
docker exec hadoop-master hdfs dfs -mkdir -p /financial_analytics/raw /financial_analytics/processed
docker exec hadoop-master hdfs dfs -put -f /opt/project/data/processed/news_cleaned.csv /financial_analytics/raw/news_cleaned.csv

# 2. Clean Previous MapReduce Output
Write-Host "`n[2/4] Cleaning previous HDFS output directory..." -ForegroundColor Yellow
docker exec hadoop-master hdfs dfs -rm -r -f /financial_analytics/processed/mr_word_count 2>&1 | Out-Null

# 3. Submit MapReduce Job to YARN
Write-Host "`n[3/4] Submitting Hadoop Streaming MapReduce job to YARN..." -ForegroundColor Yellow
$mrCmd = @'
STREAMING_JAR=$(ls /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-*.jar | head -n 1)
hadoop jar $STREAMING_JAR \
    -D mapreduce.job.name="FinancialNewsWordCount" \
    -D mapreduce.job.reduces=2 \
    -input /financial_analytics/raw/news_cleaned.csv \
    -output /financial_analytics/processed/mr_word_count \
    -mapper "python3 /opt/project/src/bigdata/mapreduce/news_mapper.py" \
    -reducer "python3 /opt/project/src/bigdata/mapreduce/news_reducer.py"
'@

docker exec hadoop-master /bin/bash -c $mrCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ MapReduce execution failed. Check YARN logs." -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ MapReduce job completed successfully on YARN!" -ForegroundColor Green

# 4. Inspect Output from HDFS
Write-Host "`n[4/4] Output files in HDFS (/financial_analytics/processed/mr_word_count):" -ForegroundColor Cyan
docker exec hadoop-master hdfs dfs -ls /financial_analytics/processed/mr_word_count

Write-Host "`nTop 15 Aggregated Results from HDFS:" -ForegroundColor Yellow
docker exec hadoop-master hdfs dfs -cat "/financial_analytics/processed/mr_word_count/part-*" | Select-Object -First 15

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " MapReduce on YARN Verification Complete ✨" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

