<#
.SYNOPSIS
    Downloads and cryptographically verifies official Apache Hadoop 3.3.6 and Spark 3.5.1
    archives into docker/cluster/dist/ for offline, deterministic Docker image builds.

.DESCRIPTION
    Academic Integrity & Rigor Protocol:
    1. Downloads official binary releases directly from Apache Software Foundation archives.
    2. Validates downloaded archives against official SHA-512 cryptographic digests.
    3. Prevents corrupted or unverified archives from entering the Docker build context.
#>

[CmdletBinding()]
param (
    [string]$DistDir = "$PSScriptRoot\..\docker\cluster\dist"
)

$ErrorActionPreference = "Stop"

# Ensure dist directory exists
if (-not (Test-Path $DistDir)) {
    New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
}

$Archives = @(
    @{
        Name         = "Hadoop 3.3.6"
        FileName     = "hadoop-3.3.6.tar.gz"
        Urls         = @(
            "https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz",
            "https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz"
        )
        ExpectedHash = "de3eaca2e0517e4b569a88b63c89fae19cb8ac6c01ff990f1ff8f0cc0f3128c8e8a23db01577ca562a0e0bb1b4a3889f8c74384e609cd55e537aada3dcaa9f8a"
    },
    @{
        Name         = "Spark 3.5.1"
        FileName     = "spark-3.5.1-bin-hadoop3.tgz"
        Urls         = @(
            "https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz"
        )
        ExpectedHash = "3d8e3f082c602027d540771e9eba9987f8ea955e978afc29e1349eb6e3f9fe32543e3d3de52dff048ebbd789730454c96447c86ff5b60a98d72620a0f082b355"
    }
)

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " Apache Cluster Distribution Staging & Verification Engine" -ForegroundColor Cyan
Write-Host " Target Directory: $DistDir" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

foreach ($item in $Archives) {
    $outPath = Join-Path $DistDir $item.FileName
    Write-Host "`n>> Inspecting $($item.Name) ($($item.FileName))..." -ForegroundColor Yellow

    $needsDownload = $true
    if (Test-Path $outPath) {
        Write-Host "   Existing archive detected. Verifying SHA-512 hash..." -ForegroundColor Gray
        $actualHash = (Get-FileHash -Path $outPath -Algorithm SHA512).Hash.ToLowerInvariant()
        if ($actualHash -eq $item.ExpectedHash.ToLowerInvariant()) {
            Write-Host "   [PASS] SHA-512 matches official Apache signature!" -ForegroundColor Green
            $needsDownload = $false
        } else {
            Write-Host "   [WARN] Hash mismatch detected! File may be incomplete or corrupt." -ForegroundColor Red
            Write-Host "   Expected: $($item.ExpectedHash)" -ForegroundColor Red
            Write-Host "   Actual:   $actualHash" -ForegroundColor Red
            Remove-Item $outPath -Force
        }
    }

    if ($needsDownload) {
        $downloadSuccess = $false
        foreach ($url in $item.Urls) {
            Write-Host "   Attempting download from: $url" -ForegroundColor Gray
            try {
                # Prefer curl.exe with resume support if available
                if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
                    & curl.exe -L -C - --retry 3 --connect-timeout 20 -o $outPath $url
                } else {
                    Invoke-WebRequest -Uri $url -OutFile $outPath -UseBasicParsing
                }

                if (Test-Path $outPath) {
                    $downloadSuccess = $true
                    break
                }
            } catch {
                Write-Host "   Download failed from $url : $_" -ForegroundColor DarkYellow
            }
        }

        if (-not $downloadSuccess -or -not (Test-Path $outPath)) {
            Write-Host "   [ERROR] Inaccessible archive: Unable to download $($item.FileName)." -ForegroundColor Red
            continue
        }

        Write-Host "   Verifying cryptographic SHA-512 hash of downloaded archive..." -ForegroundColor Gray
        $actualHash = (Get-FileHash -Path $outPath -Algorithm SHA512).Hash.ToLowerInvariant()
        if ($actualHash -eq $item.ExpectedHash.ToLowerInvariant()) {
            Write-Host "   [PASS] Successfully verified $($item.FileName)!" -ForegroundColor Green
        } else {
            Write-Host "   [FAIL] Cryptographic integrity verification failed!" -ForegroundColor Red
            Remove-Item $outPath -Force
            throw "Integrity check failed for $($item.FileName)"
        }
    }
}

Write-Host "`n==========================================================================" -ForegroundColor Cyan
Write-Host " Staging check complete." -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

