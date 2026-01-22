"""Data ingestion from nfl_data_py."""

import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

try:
    import nfl_data_py as nfl
except ImportError:
    raise ImportError("nfl_data_py is required. Install with: pip install nfl-data-py")

from .config import PARQUET_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_pbp(seasons: List[int], save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Ingest play-by-play data for given seasons.
    
    Args:
        seasons: List of seasons to ingest
        save_path: Optional path to save Parquet file
        
    Returns:
        DataFrame with play-by-play data
    """
    logger.info(f"Ingesting play-by-play data for seasons {seasons}")
    
    # Fetch play-by-play data
    pbp = nfl.import_pbp_data(years=seasons, downcast=True)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pbp.to_parquet(save_path, index=False, compression="snappy")
        logger.info(f"Saved play-by-play data to {save_path}")
    
    return pbp


def ingest_rosters(seasons: List[int], save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Ingest roster data for given seasons.
    
    Args:
        seasons: List of seasons to ingest
        save_path: Optional path to save Parquet file
        
    Returns:
        DataFrame with roster data
    """
    logger.info(f"Ingesting roster data for seasons {seasons}")
    
    rosters = nfl.import_rosters(years=seasons)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rosters.to_parquet(save_path, index=False, compression="snappy")
        logger.info(f"Saved roster data to {save_path}")
    
    return rosters


def ingest_schedules(seasons: List[int], save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Ingest schedule data for given seasons.
    
    Args:
        seasons: List of seasons to ingest
        save_path: Optional path to save Parquet file
        
    Returns:
        DataFrame with schedule data
    """
    logger.info(f"Ingesting schedule data for seasons {seasons}")
    
    schedules = nfl.import_schedules(years=seasons)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        schedules.to_parquet(save_path, index=False, compression="snappy")
        logger.info(f"Saved schedule data to {save_path}")
    
    return schedules


def ingest_all(seasons: List[int], output_dir: Optional[Path] = None) -> dict:
    """
    Ingest all data sources for given seasons.
    
    Args:
        seasons: List of seasons to ingest
        output_dir: Directory to save Parquet files (defaults to PARQUET_DIR)
        
    Returns:
        Dictionary with keys: pbp, rosters, schedules
    """
    output_dir = output_dir or PARQUET_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = {}
    
    # Ingest each data source
    data["pbp"] = ingest_pbp(seasons, output_dir / "pbp.parquet")
    data["rosters"] = ingest_rosters(seasons, output_dir / "rosters.parquet")
    data["schedules"] = ingest_schedules(seasons, output_dir / "schedules.parquet")
    
    logger.info("Ingestion complete")
    
    return data
