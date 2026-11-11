# Hadoop & Spark Cluster Deployment and Operations Guide

**Course:** CSBB 422 — Big Data Analytics Lab  
**Institution:** National Institute of Technology Delhi  
**Project:** Financial Intelligence & Big Data Analytics Platform  

---

## 1. Architecture & Deployment Model

The distributed compute and storage infrastructure is orchestrated using Docker Compose on a single workstation host.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HOST MACHINE (Windows 11 / WSL2)                      │
│                                                                             │
│  ┌────────────────────────┐  RPC :9000   ┌───────────────────────────────┐  │
│  │     hadoop-master      │◄────────────►│        hadoop-worker1         │  │
│  │  - NameNode (:9870)    │              │  - DataNode (:9864)           │  │
│  │  - ResourceManager     │  YARN :8032  │  - NodeManager (:8042)        │  │
│  │  - Spark Master (:8080)│◄────────────►│  - Spark Worker (:8081)       │  │
│  └───────────┬────────────┘              └───────────────────────────────┘  │
│              │                                           ▲                  │
│              │ RPC :9000                                 │ Block Pipeline   │
│              ▼                                           ▼ (Rep = 2)        │
│  ┌────────────────────────┐              ┌───────────────────────────────┐  │
│  │   financial-mongodb    │              │        hadoop-worker2         │  │
│  │  - mongod (:27017)     │              │  - DataNode (:9865)           │  │
│  │  - NoSQL Collections   │              │  - NodeManager (:8043)        │  │
│  └────────────────────────┘              │  - Spark Worker (:8082)       │  │
│                                          └───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Academic Statement on Single-Host Topology
> [!NOTE]
> This deployment uses three logical Hadoop/Spark nodes in separate Docker containers on one physical Windows workstation. It demonstrates true multi-process, multi-container distributed-service configuration with network-isolated daemons, but is not a physical rack-mounted cluster across three physical computers.

---

## 2. Cluster Build Specifications

- **Base Image:** `eclipse-temurin:11-jdk-focal`
- **Hadoop Version:** Apache Hadoop 3.3.6 (cryptographically verified SHA-512)
- **Spark Version:** Apache Spark 3.5.1 (bin-hadoop3, SHA-512 verified)
- **Image Tag:** `financial-analytics/hadoop-spark-cluster:3.3.6-3.5.1`
- **Replication Factor:** $R=2$ configured in `hdfs-site.xml` (`dfs.replication=2`).
- **YARN Resource Allocation:**
  - `yarn.nodemanager.resource.memory-mb`: 4,096 MB per worker.
  - `yarn.scheduler.maximum-allocation-mb`: 3,072 MB per container.

---

## 3. Essential HDFS Operations & Verification Commands

All HDFS operations are executed via the master container:

### Checking File Ingestion & Replication
```bash
docker exec -it hadoop-master hdfs dfs -ls -R /financial-intelligence/real/
```

### Inspecting HDFS Cluster Health
```bash
docker exec -it hadoop-master hdfs dfsadmin -report
```
Expected output confirms:
- **Live datanodes:** 2 (`hadoop-worker1`, `hadoop-worker2`)
- **Dead datanodes:** 0
- **Under-replicated blocks:** 0
- **Corrupt blocks:** 0

### Checking YARN NodeManager Registration
```bash
docker exec -it hadoop-master yarn node -list
```
Confirms 2 active NodeManagers with 8,192 MB total available cluster memory.

---

## 4. Submitting Distributed Spark Jobs

Distributed Spark jobs are dispatched to the Spark Master using `spark-submit`:

```bash
docker exec -it hadoop-master /opt/spark/bin/spark-submit \
  --master spark://hadoop-master:7077 \
  --deploy-mode client \
  --driver-memory 1G \
  --executor-memory 1G \
  --executor-cores 1 \
  /workspace/scripts/run_step8_cluster_analytics.py
```

### Step 8 Spark Execution Verification
- **Application ID:** `app-20261110170818-0001`
- **Status:** `FINISHED`
- **Exit Code:** `0`
- **Outputs Persisted:**
  1. `/financial-intelligence/real/analytics/risk_return_metrics.parquet`
  2. `/financial-intelligence/real/analytics/market_breadth_daily.parquet`
  3. `/financial-intelligence/real/analytics/correlation_matrix_50x50.parquet`
  4. `/financial-intelligence/real/analytics/stock_rankings.parquet`

---

## 5. Maintenance, Backup & Shutdown

### Graceful Cluster Stop
```powershell
docker compose stop
```

### Cluster Teardown (Preserving Persistent Volumes)
```powershell
docker compose down
```

### Complete Reset (Caution: Deletes HDFS NameNode Volumes)
```powershell
docker compose down -v
```

