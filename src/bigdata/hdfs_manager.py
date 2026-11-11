"""
HDFS manager — helper functions for Hadoop Distributed File System operations.

Uses the `hdfs dfs` CLI (standard Hadoop client) via subprocess.
Falls back gracefully when HDFS is unavailable, allowing the project to
work entirely in local mode for development.

Usage:
    from src.bigdata.hdfs_manager import HDFSManager
    mgr = HDFSManager(cfg["hdfs"])
    if mgr.is_available():
        mgr.create_project_dirs()
        mgr.upload_file("data/processed/market_features.csv", "/financial_analytics/raw/")
"""

import shutil
import socket
import subprocess
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import logger


class HDFSManager:
    """
    Helper class for HDFS operations using the `hdfs dfs` CLI.

    Gracefully handles the case where HDFS is unavailable by falling back
    to local storage with clear log messages.
    """

    def __init__(self, hdfs_cfg: dict):
        """
        Parameters
        ----------
        hdfs_cfg : dict
            HDFS configuration from config.yaml.
        """
        self.namenode = hdfs_cfg.get("namenode", "hdfs://localhost:9000")
        self.project_root = hdfs_cfg.get("project_root", "/financial_analytics")
        self.raw_path = hdfs_cfg.get("raw_path", "/financial_analytics/raw")
        self.processed_path = hdfs_cfg.get("processed_path", "/financial_analytics/processed")
        self._available: Optional[bool] = None

    # ─────────────────────────────────────────────────────────
    # Availability Check
    # ─────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Check if HDFS namenode is reachable.

        Performs a TCP connect to the namenode host:port and also verifies
        that the `hdfs` CLI command exists on PATH.

        Returns
        -------
        bool
            True if HDFS is reachable and the CLI is available.
        """
        if self._available is not None:
            return self._available

        # Check 1: Is the `hdfs` CLI on PATH?
        if shutil.which("hdfs") is None:
            logger.warning(
                "HDFS CLI ('hdfs') not found on PATH. "
                "HDFS operations will be skipped. "
                "Install Hadoop or use Docker for HDFS support."
            )
            self._available = False
            return False

        # Check 2: Is the namenode reachable?
        parsed = urlparse(self.namenode)
        host = parsed.hostname or "localhost"
        port = parsed.port or 9000

        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            logger.info(f"HDFS namenode reachable at {host}:{port}")
            self._available = True
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.warning(
                f"HDFS namenode not reachable at {host}:{port} ({e}). "
                f"Falling back to local storage. "
                f"Start HDFS or use Docker: docker-compose up -d"
            )
            self._available = False

        return self._available

    # ─────────────────────────────────────────────────────────
    # Internal: run hdfs dfs commands
    # ─────────────────────────────────────────────────────────

    def _run_hdfs(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """
        Run an `hdfs dfs` command.

        Parameters
        ----------
        args : list[str]
            Arguments to pass after `hdfs dfs`, e.g. ["-ls", "/path"].
        check : bool
            Whether to raise on non-zero exit code.

        Returns
        -------
        subprocess.CompletedProcess
        """
        cmd = ["hdfs", "dfs"] + args
        logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check and result.returncode != 0:
            logger.error(f"HDFS command failed: {result.stderr.strip()}")
        return result

    # ─────────────────────────────────────────────────────────
    # Directory Operations
    # ─────────────────────────────────────────────────────────

    def create_project_dirs(self) -> bool:
        """
        Create project directories in HDFS.

        Creates:
            /financial_analytics/
            /financial_analytics/raw/
            /financial_analytics/processed/

        Returns
        -------
        bool
            True if directories were created successfully.
        """
        if not self.is_available():
            logger.info("HDFS unavailable — skipping directory creation")
            return False

        dirs = [self.project_root, self.raw_path, self.processed_path]
        success = True

        for d in dirs:
            result = self._run_hdfs(["-mkdir", "-p", d], check=False)
            if result.returncode == 0:
                logger.info(f"  Created HDFS directory: {d}")
            else:
                # Directory may already exist — that's OK
                if "already exists" in result.stderr.lower():
                    logger.debug(f"  HDFS directory exists: {d}")
                else:
                    logger.error(f"  Failed to create {d}: {result.stderr.strip()}")
                    success = False

        return success

    # ─────────────────────────────────────────────────────────
    # File Operations
    # ─────────────────────────────────────────────────────────

    def upload_file(self, local_path: str, hdfs_path: str) -> bool:
        """
        Upload a local file to HDFS.

        Parameters
        ----------
        local_path : str
            Path to the local file.
        hdfs_path : str
            Destination path in HDFS.

        Returns
        -------
        bool
            True if upload succeeded.
        """
        if not self.is_available():
            logger.info(f"HDFS unavailable — skipping upload of {local_path}")
            return False

        local = Path(local_path)
        if not local.exists():
            logger.error(f"Local file not found: {local_path}")
            return False

        result = self._run_hdfs(["-put", "-f", str(local), hdfs_path], check=False)
        if result.returncode == 0:
            logger.info(f"  Uploaded: {local_path} → {hdfs_path}")
            return True
        else:
            logger.error(f"  Upload failed: {result.stderr.strip()}")
            return False

    def download_file(self, hdfs_path: str, local_path: str) -> bool:
        """
        Download a file from HDFS to local storage.

        Parameters
        ----------
        hdfs_path : str
            Source path in HDFS.
        local_path : str
            Destination path on local filesystem.

        Returns
        -------
        bool
            True if download succeeded.
        """
        if not self.is_available():
            logger.info(f"HDFS unavailable — skipping download of {hdfs_path}")
            return False

        result = self._run_hdfs(["-get", "-f", hdfs_path, local_path], check=False)
        if result.returncode == 0:
            logger.info(f"  Downloaded: {hdfs_path} → {local_path}")
            return True
        else:
            logger.error(f"  Download failed: {result.stderr.strip()}")
            return False

    def list_files(self, hdfs_path: str) -> List[str]:
        """
        List files in an HDFS directory.

        Parameters
        ----------
        hdfs_path : str
            HDFS directory to list.

        Returns
        -------
        list[str]
            File paths in the directory, or empty list if unavailable.
        """
        if not self.is_available():
            logger.info(f"HDFS unavailable — cannot list {hdfs_path}")
            return []

        result = self._run_hdfs(["-ls", hdfs_path], check=False)
        if result.returncode != 0:
            logger.warning(f"  Could not list {hdfs_path}: {result.stderr.strip()}")
            return []

        files = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("Found"):
                # HDFS -ls output format: permissions replication owner group size date time path
                parts = line.split()
                if len(parts) >= 8:
                    files.append(parts[-1])

        logger.info(f"  HDFS {hdfs_path}: {len(files)} files")
        return files

    def file_exists(self, hdfs_path: str) -> bool:
        """
        Check if a file or directory exists in HDFS.

        Parameters
        ----------
        hdfs_path : str
            Path to check.

        Returns
        -------
        bool
            True if the path exists.
        """
        if not self.is_available():
            return False

        result = self._run_hdfs(["-test", "-e", hdfs_path], check=False)
        exists = result.returncode == 0
        logger.debug(f"  HDFS exists {hdfs_path}: {exists}")
        return exists

    def get_summary(self) -> dict:
        """
        Get a summary of HDFS status and project structure.

        Returns
        -------
        dict
            Summary information.
        """
        summary = {
            "namenode": self.namenode,
            "available": self.is_available(),
            "project_root": self.project_root,
        }

        if self.is_available():
            summary["raw_files"] = self.list_files(self.raw_path)
            summary["processed_files"] = self.list_files(self.processed_path)

        return summary
