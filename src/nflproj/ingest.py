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
    
    # Try different function name variations
    if hasattr(nfl, 'import_rosters'):
        rosters = nfl.import_rosters(years=seasons)
    elif hasattr(nfl, 'import_roster'):
        rosters = nfl.import_roster(years=seasons)
    elif hasattr(nfl, 'load_rosters'):
        rosters = nfl.load_rosters(years=seasons)
    elif hasattr(nfl, 'import_players'):
        # Try different parameter names
        try:
            rosters = nfl.import_players(years=seasons)
        except TypeError:
            try:
                rosters = nfl.import_players(seasons=seasons)
            except TypeError:
                rosters = nfl.import_players()
                # Filter to requested seasons if possible
                if 'season' in rosters.columns:
                    rosters = rosters[rosters['season'].isin(seasons)]
    else:
        raise AttributeError(
            "Could not find roster import function. Available functions: " +
            str([x for x in dir(nfl) if not x.startswith('_')])
        )
    
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
    
    # Try different function name variations
    if hasattr(nfl, 'import_schedules'):
        schedules = nfl.import_schedules(years=seasons)
    elif hasattr(nfl, 'import_schedule'):
        schedules = nfl.import_schedule(years=seasons)
    elif hasattr(nfl, 'load_schedules'):
        schedules = nfl.load_schedules(years=seasons)
    else:
        # Extract schedule from pbp data (fallback)
        logger.info("Extracting schedule from pbp data")
        pbp = nfl.import_pbp_data(years=seasons, downcast=True)
        # Extract unique games from pbp
        schedule_cols = []
        for col in ['game_id', 'old_game_id', 'game_date', 'gameday', 'home_team', 'away_team', 'season', 'week', 'posteam', 'defteam']:
            if col in pbp.columns:
                schedule_cols.append(col)
        
        if 'game_id' in pbp.columns:
            # Get unique games
            schedules = pbp[schedule_cols].drop_duplicates(subset=['game_id'])
            # Ensure we have required columns
            if 'gameday' not in schedules.columns and 'game_date' in schedules.columns:
                schedules = schedules.rename(columns={'game_date': 'gameday'})
            elif 'gameday' not in schedules.columns:
                # Try to construct from other columns
                if 'season' in schedules.columns and 'week' in schedules.columns:
                    schedules['gameday'] = pd.to_datetime(schedules['season'].astype(str) + '-09-01')
                else:
                    raise ValueError("Cannot extract game dates from pbp data")
        else:
            raise AttributeError(
                "Could not find schedule import function and cannot extract from pbp. Available functions: " +
                str([x for x in dir(nfl) if not x.startswith('_')])
            )
    
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
