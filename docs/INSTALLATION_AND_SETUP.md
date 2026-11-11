# Installation, Environment Setup & Quickstart Guide

**Course:** CSBB 422 — Big Data Analytics Lab  
**Institution:** National Institute of Technology Delhi  
**Project:** Financial Intelligence & Big Data Analytics Platform  

---

## 1. System Requirements

### Workstation Prerequisites
- **Operating System:** Windows 10/11 (64-bit) with WSL2 enabled (Ubuntu/Debian recommended).
- **CPU:** 4 physical cores minimum (8 logical cores recommended).
- **RAM:** 16 GB minimum (32 GB recommended for running Hadoop containers and local ML concurrently).
- **Disk Space:** 25 GB free disk space.
- **Software Dependencies:**
  - Docker Desktop 4.28+ with WSL2 engine active.
  - Python 3.12.x (64-bit).
  - OpenJDK 11 LTS (`JAVA_HOME` configured).
  - Git for Windows.

---

## 2. Step-by-Step Installation

### Step 1: Clone Repository
```powershell
git clone <repository-url> C:\Projects\BigDataProject
cd C:\Projects\BigDataProject
```

### Step 2: Set Up Python Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 3: Verify Java 11 Environment
PySpark and Hadoop client commands require Java 11:
```powershell
java -version
# Expected: openjdk version "11.0.x"
```

---

## 3. Launching the Containerized Cluster

The cluster consists of three Hadoop/Spark containers and one MongoDB container managed by Docker Compose.

### Cluster Startup
```powershell
docker compose up -d
```

### Health & Container Verification
Verify all 4 containers report healthy states:
```powershell
docker compose ps
```
Expected output:
```text
NAME                IMAGE                                              STATUS
financial-mongodb   mongo:6.0                                          Up (healthy)
hadoop-master       financial-analytics/hadoop-spark-cluster:3.3.6-3.5.1 Up (healthy)
hadoop-worker1      financial-analytics/hadoop-spark-cluster:3.3.6-3.5.1 Up (healthy)
hadoop-worker2      financial-analytics/hadoop-spark-cluster:3.3.6-3.5.1 Up (healthy)
```

### Accessing Cluster Web Consoles
- **HDFS NameNode UI:** [http://localhost:9870](http://localhost:9870)
- **Hadoop YARN ResourceManager:** [http://localhost:8088](http://localhost:8088)
- **Spark Master UI:** [http://localhost:8080](http://localhost:8080)
- **Spark Worker 1 UI:** [http://localhost:8081](http://localhost:8081)
- **Spark Worker 2 UI:** [http://localhost:8082](http://localhost:8082)
- **Hadoop JobHistory Server:** [http://localhost:19888](http://localhost:19888)

---

## 4. Launching the Interactive Terminal UI

Start the Streamlit dark-themed financial dashboard:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app/Home.py
```
- Open your browser at [http://localhost:8501](http://localhost:8501).
- The sidebar displays active cluster health, active dataset context switcher, and links to all 11 intelligence modules.

---

## 5. Running the Complete Verification Test Suite

The project includes an end-to-end automated test suite covering unit, integration, and distributed jobs.

### Running Standard Test Suite (276 Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest tests --ignore=tests/test_spark_jobs.py -v
```

### Running Real ML Tests (6 Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_real_ml_pipeline.py -v
```

### Running PySpark Regression Suite (8 Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_spark_jobs.py -v
```
*(All 284 tests pass with zero failures and zero errors).*

