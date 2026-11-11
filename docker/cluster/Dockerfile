# ==============================================================================
# Financial Intelligence Platform — Multi-Node Hadoop & Spark Node Image
# Pinned Versions:
#   - Java:   OpenJDK 11 (Eclipse Temurin Jammy)
#   - Hadoop: Apache Hadoop 3.3.6
#   - Spark:  Apache Spark 3.5.1 (bin-hadoop3)
#   - Python: Python 3.10+ (Ubuntu Jammy default)
# ==============================================================================

FROM eclipse-temurin:11-jdk-jammy

LABEL maintainer="Antigravity Financial Intelligence Platform"
LABEL version="1.0.0"
LABEL description="Unified Apache Hadoop 3.3.6 and Apache Spark 3.5.1 Cluster Node"

ENV DEBIAN_FRONTEND=noninteractive

# Install essential Linux tools, networking utilities, and Python 3
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tar \
    sudo \
    procps \
    netcat-openbsd \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Environment Variables
ENV JAVA_HOME=/opt/java/openjdk
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV HADOOP_COMMON_HOME=/opt/hadoop
ENV HADOOP_HDFS_HOME=/opt/hadoop
ENV HADOOP_YARN_HOME=/opt/hadoop
ENV HADOOP_MAPRED_HOME=/opt/hadoop
ENV SPARK_HOME=/opt/spark
ENV SPARK_CONF_DIR=/opt/spark/conf
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV PATH=$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# Pinned Software Versions
ARG HADOOP_VERSION=3.3.6
ARG SPARK_VERSION=3.5.1

# Install Apache Hadoop 3.3.6 from locally staged, cryptographically verified archive
COPY dist/hadoop-3.3.6.tar.gz /tmp/hadoop.tar.gz
RUN echo "Extracting Apache Hadoop ${HADOOP_VERSION}..." && \
    tar -xzf /tmp/hadoop.tar.gz -C /opt/ && \
    mv "/opt/hadoop-${HADOOP_VERSION}" /opt/hadoop && \
    rm /tmp/hadoop.tar.gz

# Install Apache Spark 3.5.1 from locally staged, cryptographically verified archive
COPY dist/spark-3.5.1-bin-hadoop3.tgz /tmp/spark.tar.gz
RUN echo "Extracting Apache Spark ${SPARK_VERSION}..." && \
    tar -xzf /tmp/spark.tar.gz -C /opt/ && \
    mv "/opt/spark-${SPARK_VERSION}-bin-hadoop3" /opt/spark && \
    rm /tmp/spark.tar.gz

# Create Data and Log Directories
RUN mkdir -p /opt/hadoop/data/nameNode \
             /opt/hadoop/data/dataNode \
             /opt/hadoop/tmp \
             /opt/hadoop/logs \
             /opt/spark/logs \
             /opt/project

# Install required Python packages for in-cluster PySpark jobs
RUN pip3 install --no-cache-dir \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    pyarrow>=14.0.0

# Copy Hadoop XML Configuration Files
COPY config/core-site.xml /opt/hadoop/etc/hadoop/core-site.xml
COPY config/hdfs-site.xml /opt/hadoop/etc/hadoop/hdfs-site.xml
COPY config/yarn-site.xml /opt/hadoop/etc/hadoop/yarn-site.xml
COPY config/mapred-site.xml /opt/hadoop/etc/hadoop/mapred-site.xml
COPY config/workers /opt/hadoop/etc/hadoop/workers

# Copy Spark Configuration Files
COPY config/spark-defaults.conf /opt/spark/conf/spark-defaults.conf

# Copy Entrypoint Script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /opt/project

ENTRYPOINT ["/entrypoint.sh"]

