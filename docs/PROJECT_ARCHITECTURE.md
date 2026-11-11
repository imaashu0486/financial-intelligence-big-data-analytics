# Distributed System Architecture & Infrastructure Topology

**Course:** CSBB 422 — Big Data Analytics Lab  
**Institution:** National Institute of Technology Delhi  
**Project:** Financial Intelligence & Big Data Analytics Platform  
**Target Market:** National Stock Exchange of India (NSE) — NIFTY 50 Index  

---

## 1. Overview & Academic Topology Statement

The **Financial Intelligence & Big Data Analytics Platform** is an academic big data processing, financial econometrics, and machine learning platform designed to ingest, process, model, and visualize 10 years of Indian equity market data (2015–2024).

### Academic Hardware & Cluster Reality Disclosure
> [!IMPORTANT]
> **Single-Host Containerized Logical Cluster:**  
> For academic reproducibility and laboratory evaluation, the distributed storage and processing layers are implemented as a **three-node logical cluster hosted inside Docker containers on a single physical host machine** (Windows 11 with WSL2 Linux subsystem backend).
> 
> - It is **NOT** a cluster running across three discrete physical hardware servers.
> - However, it executes **genuine distributed daemons** (`NameNode`, `DataNode`, `ResourceManager`, `NodeManager`, `SparkMaster`, `SparkWorker`) across separate container network boundaries with configured replication ($R=2$), YARN resource isolation, and remote RPC protocols.

---

## 2. End-to-End System Topology

```mermaid
flowchart TD
    subgraph INGESTION ["1. Ingestion & Preprocessing Layer"]
        A1["Real NIFTY 50 Equities<br/>(121,962 daily OHLCV rows)"]
        A2["NIFTY 50 Benchmark (^NSEI)<br/>(2,459 trading sessions)"]
        A3["Corporate News & Filings<br/>(79 articles [67 aligned] + 4 MD&A extracts)"]
        A4["Local Preprocessing Engine<br/>(14 Technical Features + Target Isolation)"]
        A1 --> A4
        A2 --> A4
    end

    subgraph STORAGE ["2. Distributed Storage Layer (HDFS)"]
        B1["hadoop-master<br/>(NameNode :9870)"]
        B2["hadoop-worker1<br/>(DataNode 1 :9864)"]
        B3["hadoop-worker2<br/>(DataNode 2 :9865)"]
        B1 --- B2
        B1 --- B3
        A4 -.->|"HDFS Put (Replication=2)"| B1
    end

    subgraph COMPUTE ["3. Distributed Compute Engine"]
        C1["Hadoop YARN 3.3.6<br/>(ResourceManager :8088)"]
        C2["Apache Spark 3.5.1 Cluster<br/>(Master :8080 | 2 Executors)"]
        B1 --> C1
        B1 --> C2
        C2 -->|"Distributed Execution<br/>(app-20261110170818-0001)"| D_TABLES
    end

    subgraph D_TABLES ["4. Analytical Data Products (Parquet & HDFS)"]
        T1["risk_return_metrics.parquet (50 stocks)"]
        T2["market_breadth_daily.parquet (2,459 days)"]
        T3["correlation_matrix_50x50.parquet (2,500 pairs)"]
        T4["stock_rankings.parquet (Multi-Factor Score)"]
    end

    subgraph ML_NLP ["5. Machine Learning & NLP Engines"]
        M1["Real ML Engine<br/>(LogReg, Ridge, RF, GBDT)"]
        M2["Financial NLP Lexicon<br/>(Directional News Sentiment)"]
        M3["Financial Document RAG<br/>(all-MiniLM-L6-v2 + Extractive QA)"]
    end

    subgraph PERSISTENCE ["6. Persistence & Serving Layer"]
        P1["MongoDB NoSQL Cluster<br/>(financial-mongodb :27017)"]
        P2["Local Model Registry<br/>(models/real/*.joblib)"]
    end

    subgraph UI ["7. Interactive Terminal Interface"]
        U1["Streamlit Financial Terminal<br/>(11 Multipage Interactive Modules)"]
        U2["Plotly High-Resolution Dashboards"]
    end

    D_TABLES --> P1
    D_TABLES --> M1
    M1 --> P2
    P1 --> U1
    P2 --> U1
    T1 & T2 & T3 & T4 --> U1
    M3 --> U1
    U1 --> U2
```

---

## 3. Containerized Node Specifications

The distributed services run on a dedicated Docker bridge network (`bigdataproject_default` / `172.18.0.0/16`):

| Node / Container Name | Service Roles & Daemons | Memory Limit | Exposed Ports | Function in Platform |
| :--- | :--- | :--- | :--- | :--- |
| **`hadoop-master`** | `NameNode`<br/>`ResourceManager`<br/>`Spark Master`<br/>`JobHistoryServer` | 4,096 MB | `9870` (NameNode UI)<br/>`8088` (YARN UI)<br/>`8080` (Spark Master UI)<br/>`19888` (JobHistory)<br/>`7077` (Spark RPC) | Orchestrates block storage, YARN application master scheduling, and distributed Spark DAG dispatching. |
| **`hadoop-worker1`** | `DataNode`<br/>`NodeManager`<br/>`Spark Worker` | 3,072 MB | `9864` (DataNode 1)<br/>`8042` (NodeManager 1)<br/>`8081` (Spark Worker 1) | Hosts Block Replica 1, executes YARN mapper/reducer containers, and runs Spark Executor 1 (1 core, 1 GB). |
| **`hadoop-worker2`** | `DataNode`<br/>`NodeManager`<br/>`Spark Worker` | 3,072 MB | `9865` (DataNode 2)<br/>`8043` (NodeManager 2)<br/>`8082` (Spark Worker 2) | Hosts Block Replica 2, executes YARN mapper/reducer containers, and runs Spark Executor 2 (1 core, 1 GB). |
| **`financial-mongodb`** | `mongod` 6.0 | 2,048 MB | `27017` (MongoDB Wire Protocol) | Persists enriched financial metrics, corporate entities, stock rankings, and user session cache. |

---

## 4. Storage Architecture (HDFS & Local Parquet)

### HDFS Namespace Layout (`/financial-intelligence/real/`)
- `/financial-intelligence/real/market/`
  - `consolidated_market_features.parquet` (26.2 MB): 121,962 rows across 50 equities.
  - `benchmark_features.parquet` (294 KB): 2,459 official NIFTY 50 benchmark sessions.
  - `daily_features/`: 50 individual equity parquet files.
- `/financial-intelligence/real/analytics/`
  - `risk_return_metrics.parquet`: 50 equities with CAGR, annualized volatility, Sharpe ($R_f=6.0\%$), CAPM Beta, Alpha, Max Drawdown.
  - `market_breadth_daily.parquet`: 2,459 sessions with Advance/Decline ratios and Moving Average participation percentages.
  - `correlation_matrix_50x50.parquet`: 2,500 pairwise daily return correlation cells.
  - `stock_rankings.parquet`: 50 equities ranked by multi-factor composite investment score.
- `/financial-intelligence/real/news/`
  - `cleaned_financial_news.csv` (79 corporate disclosure records; 67 aligned with trading sessions).
  - `daily_company_news_features.csv`.
- `/financial-intelligence/real/documents/`
  - 4 verified FY24 annual report MD&A extract files.

---

## 5. Compute Engine & Execution Workflow

1. **Ingestion & Feature Engine (`src/features/`):**
   - Calculates 14 stationary technical indicators including log returns, momentum, rolling annualized volatility ($\sqrt{252}$ scaling), moving average ratios (`close_to_ma20/50/200`), and benchmark-relative returns.
   - Applies strict anti-leakage isolation: targets are constructed via strict $t+1$ lead shifts and dropped from feature sets.
2. **Distributed Spark Processing (`src/analytics/`):**
   - PySpark jobs submit DAGs to `spark://hadoop-master:7077`.
   - Data is partitioned across worker nodes using hash partitioning on equity `symbol` and `date`.
   - Spark executes map-side aggregations, broadcast joins for benchmark metrics, and parallelized covariance matrices.
3. **Machine Learning Pipeline (`src/ml/`):**
   - Models are trained on out-of-sample chronological date partitions (Train 2015–2021, Val 2022–2023, Test 2024).
   - Logistic Regression, Ridge Regression, Random Forest, and Gradient Boosting models persist in `models/real/`.
4. **Retrieval-Augmented Generation (`src/rag/`):**
   - SentenceTransformer (`all-MiniLM-L6-v2`) generates 384-dimensional dense vectors.
   - Vector cosine retrieval is combined with lexical keyword matching and confidence-thresholded extractive question answering.
5. **Presentation Layer (`app/`):**
   - Streamlit terminal interface with 11 specialized dashboards.
   - Dual-scope switcher allows instant toggling between Real NIFTY 50 Market History and Synthetic Baseline without data contamination.
