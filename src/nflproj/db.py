"""DuckDB connection and helper functions."""

import duckdb
from pathlib import Path
from typing import Optional

from .config import DB_PATH


class DB:
    """DuckDB connection manager."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self.conn is None:
            self.conn = duckdb.connect(str(self.db_path))
        return self.conn

    def close(self):
        """Close database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def execute(self, query: str, parameters: Optional[dict] = None):
        """Execute a query."""
        conn = self.connect()
        if parameters:
            return conn.execute(query, parameters)
        return conn.execute(query)

    def query(self, query: str, parameters: Optional[dict] = None) -> duckdb.DuckDBPyRelation:
        """Execute a query and return relation."""
        conn = self.connect()
        if parameters:
            return conn.query(query, parameters)
        return conn.query(query)

    def register_parquet(self, name: str, path: str):
        """Register a Parquet file or directory as a table."""
        conn = self.connect()
        conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{path}')")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_schema(db: DB):
    """Create core schema views/tables from Parquet files."""
    parquet_dir = Path(db.db_path).parent.parent / "parquet"
    
    # Register Parquet files as tables
    for table in ["games", "players", "player_game_offense", "team_game_offense", 
                  "features_player_game_offense"]:
        parquet_path = parquet_dir / f"{table}.parquet"
        if parquet_path.exists():
            db.register_parquet(table, str(parquet_path))
