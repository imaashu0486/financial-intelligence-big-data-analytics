#!/bin/bash
set -e

# ==============================================================================
# Financial Analytics Platform — Hadoop & Spark Node Entrypoint
# Dispatches services based on NODE_TYPE: 'master' or 'worker'
# ==============================================================================

export JAVA_HOME=${JAVA_HOME:-/opt/java/openjdk}
export HADOOP_HOME=${HADOOP_HOME:-/opt/hadoop}
export HADOOP_CONF_DIR=${HADOOP_CONF_DIR:-/opt/hadoop/etc/hadoop}
export SPARK_HOME=${SPARK_HOME:-/opt/spark}
export PATH=$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

echo "======================================================================"
echo " Starting Node: $HOSTNAME (Role: ${NODE_TYPE:-master})"
echo " Java: $($JAVA_HOME/bin/java -version 2>&1 | head -n 1)"
echo " Hadoop: $HADOOP_HOME"
echo " Spark:  $SPARK_HOME"
echo "======================================================================"

if [ "${NODE_TYPE}" = "master" ]; then
    echo ">> [1/5] Checking NameNode formatting..."
    if [ ! -d "/opt/hadoop/data/nameNode/current" ]; then
        echo ">> Formatting HDFS NameNode (clusterId: financial-analytics-cluster)..."
        $HADOOP_HOME/bin/hdfs namenode -format -force -clusterId financial-analytics-cluster
    else
        echo ">> NameNode already formatted. Retaining existing metadata."
    fi

    echo ">> [2/5] Starting HDFS NameNode daemon..."
    $HADOOP_HOME/bin/hdfs --daemon start namenode

    echo ">> [3/5] Starting YARN ResourceManager daemon..."
    $HADOOP_HOME/bin/yarn --daemon start resourcemanager

    echo ">> [4/5] Starting MapReduce JobHistoryServer daemon..."
    $HADOOP_HOME/bin/mapred --daemon start historyserver

    echo ">> [5/5] Starting Apache Spark Master..."
    $SPARK_HOME/sbin/start-master.sh -h 0.0.0.0 -p 7077 --webui-port 8080

    echo ">> Waiting for HDFS to become active and leave safe mode..."
    sleep 3
    $HADOOP_HOME/bin/hdfs dfsadmin -safemode wait || true

    echo ">> Initializing standard HDFS directories..."
    $HADOOP_HOME/bin/hdfs dfs -mkdir -p /tmp /financial_analytics/raw /financial_analytics/processed || true
    $HADOOP_HOME/bin/hdfs dfs -chmod -R 777 /tmp /financial_analytics || true

    echo "======================================================================"
    echo " Hadoop Master services initialized successfully!"
    echo " NameNode Web UI:        http://localhost:9870"
    echo " YARN ResourceManager:   http://localhost:8088"
    echo " Spark Master Web UI:    http://localhost:8080"
    echo " JobHistory Server:      http://localhost:19888"
    echo "======================================================================"

elif [ "${NODE_TYPE}" = "worker" ]; then
    echo ">> Waiting for hadoop-master NameNode (port 9000) to become available..."
    while ! nc -z hadoop-master 9000; do
        sleep 2
    done
    echo ">> NameNode is reachable."

    echo ">> Waiting for hadoop-master ResourceManager (port 8032) to become available..."
    while ! nc -z hadoop-master 8032; do
        sleep 2
    done
    echo ">> ResourceManager is reachable."

    echo ">> [1/3] Starting HDFS DataNode daemon..."
    $HADOOP_HOME/bin/hdfs --daemon start datanode

    echo ">> [2/3] Starting YARN NodeManager daemon..."
    $HADOOP_HOME/bin/yarn --daemon start nodemanager

    echo ">> [3/3] Starting Apache Spark Worker..."
    $SPARK_HOME/sbin/start-worker.sh spark://hadoop-master:7077 --webui-port 8081

    echo "======================================================================"
    echo " Hadoop Worker ($HOSTNAME) services initialized successfully!"
    echo " DataNode Web UI:    http://localhost:9864 (or mapped port)"
    echo " NodeManager Web UI: http://localhost:8042 (or mapped port)"
    echo " Spark Worker UI:    http://localhost:8081 (or mapped port)"
    echo "======================================================================"
else
    echo "ERROR: Unknown NODE_TYPE '${NODE_TYPE}'. Must be 'master' or 'worker'."
    exit 1
fi

# Keep container running and stream logs
echo ">> Streaming daemon logs..."
tail -f $HADOOP_HOME/logs/*.log $SPARK_HOME/logs/*.log 2>/dev/null || tail -f /dev/null

