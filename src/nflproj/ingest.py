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
    
    # Filter out future seasons that don't exist yet
    from datetime import datetime
    current_year = datetime.now().year
    valid_seasons = [s for s in seasons if s <= current_year]
    invalid_seasons = [s for s in seasons if s > current_year]
    
    if invalid_seasons:
        logger.warning(f"Seasons {invalid_seasons} are in the future and data is not available yet. Skipping.")
    
    if not valid_seasons:
        raise ValueError(f"No valid seasons to ingest. All requested seasons ({seasons}) are in the future.")
    
    # Fetch play-by-play data
    try:
        pbp = nfl.import_pbp_data(years=valid_seasons, downcast=True)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error ingesting PBP data: {error_msg}")
        
        # Check if it's the participation data issue (common for future seasons)
        if "Error" in error_msg or "404" in error_msg or "Not Found" in error_msg:
            # Try without participation data (for seasons where it's not available)
            try:
                logger.info("Retrying without participation data...")
                pbp = nfl.import_pbp_data(years=valid_seasons, downcast=True, include_participation=False)
            except Exception as e3:
                # Try without downcast as well
                try:
                    pbp = nfl.import_pbp_data(years=valid_seasons, downcast=False, include_participation=False)
                except Exception as e4:
                    logger.error(f"Failed to ingest PBP data: {e4}")
                    raise ValueError(
                        f"Failed to ingest PBP data for seasons {valid_seasons}. "
                        f"This may be because the season data is incomplete or not yet available. "
                        f"Error: {str(e4)}"
                    )
        else:
            # Try without downcast if that fails
            try:
                pbp = nfl.import_pbp_data(years=valid_seasons, downcast=False)
            except Exception as e2:
                logger.error(f"Failed to ingest PBP data even without downcast: {e2}")
                raise ValueError(
                    f"Failed to ingest PBP data for seasons {valid_seasons}. "
                    f"Error: {str(e2)}"
                )
    
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


def ingest_all(seasons: List[int], output_dir: Optional[Path] = None, append: bool = True) -> dict:
    """
    Ingest all data sources for given seasons.
    
    Args:
        seasons: List of seasons to ingest
        output_dir: Directory to save Parquet files (defaults to PARQUET_DIR)
        append: If True, merge with existing data. If False, overwrite.
        
    Returns:
        Dictionary with keys: pbp, rosters, schedules
    """
    output_dir = output_dir or PARQUET_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = {}
    
    # Ingest each data source
    pbp_path = output_dir / "pbp.parquet"
    rosters_path = output_dir / "rosters.parquet"
    schedules_path = output_dir / "schedules.parquet"
    
    # Ingest new data
    new_pbp = ingest_pbp(seasons, None)  # Don't save yet
    new_rosters = ingest_rosters(seasons, None)
    new_schedules = ingest_schedules(seasons, None)
    
    # Merge with existing data if append=True and files exist
    if append and pbp_path.exists():
        logger.info("Merging with existing data...")
        existing_pbp = pd.read_parquet(pbp_path)
        existing_rosters = pd.read_parquet(rosters_path) if rosters_path.exists() else pd.DataFrame()
        existing_schedules = pd.read_parquet(schedules_path) if schedules_path.exists() else pd.DataFrame()
        
        # Remove duplicates (keep new data if there's overlap)
        if len(existing_pbp) > 0:
            # Remove existing seasons from old data
            if 'season' in existing_pbp.columns:
                existing_pbp = existing_pbp[~existing_pbp['season'].isin(seasons)]
            # Combine
            data["pbp"] = pd.concat([existing_pbp, new_pbp], ignore_index=True)
        else:
            data["pbp"] = new_pbp
        
        if len(existing_rosters) > 0:
            # Remove duplicates by player_id and season if available
            if 'season' in existing_rosters.columns and 'player_id' in existing_rosters.columns:
                existing_rosters = existing_rosters[
                    ~(existing_rosters['season'].isin(seasons) & 
                      existing_rosters['player_id'].isin(new_rosters.get('player_id', pd.Series()).unique()))
                ]
            data["rosters"] = pd.concat([existing_rosters, new_rosters], ignore_index=True)
        else:
            data["rosters"] = new_rosters
        
        if len(existing_schedules) > 0:
            # Remove duplicates by game_id
            if 'game_id' in existing_schedules.columns:
                existing_schedules = existing_schedules[~existing_schedules['game_id'].isin(new_schedules.get('game_id', pd.Series()).unique())]
            data["schedules"] = pd.concat([existing_schedules, new_schedules], ignore_index=True)
        else:
            data["schedules"] = new_schedules
    else:
        # No existing data or append=False, use new data only
        data["pbp"] = new_pbp
        data["rosters"] = new_rosters
        data["schedules"] = new_schedules
    
    # Save merged data
    data["pbp"].to_parquet(pbp_path, index=False, compression="snappy")
    logger.info(f"Saved play-by-play data to {pbp_path} ({len(data['pbp'])} rows)")
    
    data["rosters"].to_parquet(rosters_path, index=False, compression="snappy")
    logger.info(f"Saved roster data to {rosters_path} ({len(data['rosters'])} rows)")
    
    data["schedules"].to_parquet(schedules_path, index=False, compression="snappy")
    logger.info(f"Saved schedule data to {schedules_path} ({len(data['schedules'])} rows)")
    
    logger.info("Ingestion complete")
    
    return data
