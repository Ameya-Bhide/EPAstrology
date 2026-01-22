"""Configuration for data directories and default settings."""

from pathlib import Path

# Base directory for the project
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_DIR = DATA_DIR / "parquet"
DB_DIR = DATA_DIR / "db"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
PARQUET_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# Default seasons to ingest
DEFAULT_SEASONS = list(range(2018, 2024))  # 2018-2023

# DuckDB database path
DB_PATH = DB_DIR / "nflproj.duckdb"

# Shrinkage parameters (k values for empirical Bayes)
SHRINKAGE_K = {
    "targets": {"WR": 30, "TE": 30, "RB": 15, "default": 25},
    "carries": {"RB": 40, "WR": 10, "TE": 5, "default": 30},
}

# Rolling window sizes for features
ROLLING_WINDOWS = [1, 3, 6]
