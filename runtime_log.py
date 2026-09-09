"""
runtime_log.py

Shared runtime logging for the three selection/modelling pathways.
Each script times its core computation and appends one row per stage.
"""

import csv
import os
import platform
import time
from contextlib import contextmanager
from datetime import datetime

from config import RUNTIME_LOG   # add to config.py, e.g. DATA_DIR / "runtime_log.csv"

FIELDS = ['run_timestamp', 'method', 'stage', 'seconds',
          'machine', 'cpu_count']


def log_runtime(method, stage, seconds):
    """Append one timing row to the shared runtime log."""
    row = {
        'run_timestamp': datetime.now().isoformat(timespec='seconds'),
        'method': method,                       # 'granger' | 'lasso' | 'rf'
        'stage': stage,                         # 'selection' | 'fit_baseline' | ...
        'seconds': round(seconds, 3),
        'machine': platform.node(),
        'cpu_count': os.cpu_count(),
    }
    file_exists = os.path.exists(RUNTIME_LOG)
    with open(RUNTIME_LOG, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@contextmanager
def timed(method, stage):
    """Context manager: times the enclosed block and logs it automatically."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log_runtime(method, stage, elapsed)
        print(f"[runtime] {method}/{stage}: {elapsed:,.2f}s")