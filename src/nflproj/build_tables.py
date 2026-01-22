"""Build core tables from ingested data."""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from .config import PARQUET_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_games_table(
    schedules: pd.DataFrame,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build games table from schedules.
    
    Schema: game_id, season, week, date, home_team, away_team
    """
    logger.info("Building games table")
    
    games = schedules[[
        "game_id", "season", "week", "gameday", "home_team", "away_team"
    ]].copy()
    
    games = games.rename(columns={"gameday": "date"})
    
    # Ensure date is datetime
    games["date"] = pd.to_datetime(games["date"])
    
    # Sort by date
    games = games.sort_values(["season", "week", "date"]).reset_index(drop=True)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        games.to_parquet(save_path, index=False, compression="snappy")
        logger.info(f"Saved games table to {save_path}")
    
    return games


def build_players_table(
    rosters: pd.DataFrame,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build players table from rosters.
    
    Schema: player_id, name, position
    Uses the most recent roster entry for each player to get current position.
    """
    logger.info("Building players table")
    
    # Map column names from nfl_data_py to our expected names
    # nfl_data_py uses: gsis_id (or player_id), display_name, position, season
    column_mapping = {}
    
    # Find player ID column
    if "player_id" in rosters.columns:
        player_id_col = "player_id"
    elif "gsis_id" in rosters.columns:
        player_id_col = "gsis_id"
    else:
        raise ValueError(f"Could not find player ID column. Available columns: {list(rosters.columns)}")
    
    # Find name column
    if "player_name" in rosters.columns:
        name_col = "player_name"
    elif "display_name" in rosters.columns:
        name_col = "display_name"
    elif "name" in rosters.columns:
        name_col = "name"
    else:
        raise ValueError(f"Could not find name column. Available columns: {list(rosters.columns)}")
    
    # Find position column
    if "position" not in rosters.columns:
        raise ValueError(f"Could not find position column. Available columns: {list(rosters.columns)}")
    
    # Find season column (optional)
    season_col = "season" if "season" in rosters.columns else None
    
    # Create a working copy with standardized column names
    rosters_work = rosters.copy()
    rosters_work["_player_id"] = rosters_work[player_id_col]
    rosters_work["_name"] = rosters_work[name_col]
    rosters_work["_position"] = rosters_work["position"]
    
    if season_col:
        rosters_work["_season"] = rosters_work[season_col]
    else:
        # If no season column, assume all are from the same season
        rosters_work["_season"] = rosters_work.get("season", 2023)
    
    # Get unique players with their most recent position
    agg_dict = {
        "_name": "first",
        "_position": "last",
    }
    if season_col:
        agg_dict["_season"] = "max"
    
    players = rosters_work.groupby("_player_id").agg(agg_dict).reset_index()
    
    players = players.rename(columns={
        "_player_id": "player_id",
        "_name": "name",
        "_position": "position"
    })
    
    # Select only relevant columns
    players = players[["player_id", "name", "position"]].copy()
    
    # Remove duplicates and sort
    players = players.drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        players.to_parquet(save_path, index=False, compression="snappy")
        logger.info(f"Saved players table to {save_path}")
    
    return players


def build_player_game_offense_table(
    pbp: pd.DataFrame,
    rosters: pd.DataFrame,
    games: pd.DataFrame,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build player_game_offense table from play-by-play data.
    
    Schema: game_id, player_id, team, opponent,
            targets, receptions, receiving_yards,
            carries, rushing_yards,
            epa_total, epa_per_target, epa_per_rush,
            success_rate, redzone_touches
    """
    logger.info("Building player_game_offense table")
    
    # Filter to offensive plays with EPA
    offense_plays = pbp[
        (pbp["epa"].notna()) &
        (pbp["play_type"].isin(["pass", "run"]))
    ].copy()
    
    # Get player-team mapping from rosters (for each game, use roster from that season)
    # We'll use the team from the play itself for now
    player_stats = []
    
    # Find the correct column names for player IDs
    # nfl_data_py may use different column names
    target_id_col = None
    rusher_id_col = None
    passer_id_col = None
    
    for col in ["target_player_id", "receiver_player_id", "receiver_id", "target_id"]:
        if col in offense_plays.columns:
            target_id_col = col
            break
    
    for col in ["rusher_player_id", "rusher_id", "carrier_id"]:
        if col in offense_plays.columns:
            rusher_id_col = col
            break
    
    for col in ["passer_player_id", "passer_id", "qb_player_id"]:
        if col in offense_plays.columns:
            passer_id_col = col
            break
    
    if target_id_col is None:
        logger.warning("Could not find target player ID column. Available columns: " + 
                      str([c for c in offense_plays.columns if 'target' in c.lower() or 'receiver' in c.lower()]))
        # Try to use receiver from names if available
        if "receiver_player_name" in offense_plays.columns:
            logger.info("Using receiver_player_name to identify targets")
            target_id_col = "receiver_player_name"
        else:
            raise ValueError("Could not find target/receiver player identifier in pbp data")
    
    if rusher_id_col is None:
        logger.warning("Could not find rusher player ID column. Available columns: " + 
                      str([c for c in offense_plays.columns if 'rush' in c.lower() or 'carrier' in c.lower()]))
        # Try to use rusher from names if available
        if "rusher_player_name" in offense_plays.columns:
            logger.info("Using rusher_player_name to identify rushers")
            rusher_id_col = "rusher_player_name"
        else:
            raise ValueError("Could not find rusher/carrier player identifier in pbp data")
    
    if passer_id_col is None:
        logger.warning("Could not find passer player ID column. Available columns: " + 
                      str([c for c in offense_plays.columns if 'passer' in c.lower() or 'qb' in c.lower()]))
        # Try to use passer from names if available
        if "passer_player_name" in offense_plays.columns:
            logger.info("Using passer_player_name to identify passers")
            passer_id_col = "passer_player_name"
        else:
            # QB stats are optional - don't fail if not found
            logger.info("QB stats will not be included (passer ID column not found)")
            passer_id_col = None
    
    # Passing stats (targets, receptions, receiving yards, EPA)
    pass_plays = offense_plays[offense_plays["play_type"] == "pass"].copy()
    
    # Targets - filter to plays with a target
    if target_id_col in pass_plays.columns:
        pass_plays_with_target = pass_plays[pass_plays[target_id_col].notna()].copy()
    else:
        pass_plays_with_target = pass_plays.copy()
    
    if len(pass_plays_with_target) > 0:
        targets = pass_plays_with_target.groupby(
            ["game_id", target_id_col, "posteam"]
        ).agg({
            "epa": "sum",
            "complete_pass": "sum",
            "yards_gained": "sum",
            "success": "sum",
            "play_id": "count"  # Count as targets
        }).reset_index()
        
        targets = targets.rename(columns={
            target_id_col: "player_id",
            "posteam": "team",
            "play_id": "targets",
            "complete_pass": "receptions",
            "yards_gained": "receiving_yards",
            "epa": "epa_receiving",
            "success": "success_receiving"
        })
    else:
        # Create empty dataframe with correct structure
        targets = pd.DataFrame(columns=["game_id", "player_id", "team", "targets", 
                                        "receptions", "receiving_yards", "epa_receiving", "success_receiving"])
    
    # Rushing stats (carries, rushing yards, EPA)
    rush_plays = offense_plays[offense_plays["play_type"] == "run"].copy()
    
    # Carries - filter to plays with a rusher
    if rusher_id_col in rush_plays.columns:
        rush_plays_with_rusher = rush_plays[rush_plays[rusher_id_col].notna()].copy()
    else:
        rush_plays_with_rusher = rush_plays.copy()
    
    if len(rush_plays_with_rusher) > 0:
        carries = rush_plays_with_rusher.groupby(
            ["game_id", rusher_id_col, "posteam"]
        ).agg({
            "epa": "sum",
            "yards_gained": "sum",
            "success": "sum",
            "play_id": "count"  # Count as carries
        }).reset_index()
        
        carries = carries.rename(columns={
            rusher_id_col: "player_id",
            "posteam": "team",
            "play_id": "carries",
            "yards_gained": "rushing_yards",
            "epa": "epa_rushing",
            "success": "success_rushing"
        })
    else:
        # Create empty dataframe with correct structure
        carries = pd.DataFrame(columns=["game_id", "player_id", "team", "carries",
                                       "rushing_yards", "epa_rushing", "success_rushing"])
    
    # QB passing stats (pass attempts, completions, passing yards, TDs, INTs, EPA)
    qb_stats = pd.DataFrame()
    if passer_id_col and passer_id_col in pass_plays.columns:
        pass_plays_with_passer = pass_plays[pass_plays[passer_id_col].notna()].copy()
        
        if len(pass_plays_with_passer) > 0:
            qb_stats = pass_plays_with_passer.groupby(
                ["game_id", passer_id_col, "posteam"]
            ).agg({
                "play_id": "count",  # Pass attempts
                "complete_pass": "sum",  # Completions
                "yards_gained": "sum",  # Passing yards
                "touchdown": lambda x: (x == 1).sum(),  # Passing TDs
                "interception": lambda x: (x == 1).sum(),  # Interceptions
                "epa": "sum",  # EPA from passing
                "success": "sum"  # Successful passes
            }).reset_index()
            
            qb_stats = qb_stats.rename(columns={
                passer_id_col: "player_id",
                "posteam": "team",
                "play_id": "pass_attempts",
                "complete_pass": "completions",
                "yards_gained": "passing_yards",
                "touchdown": "passing_tds",
                "interception": "interceptions",
                "epa": "epa_passing",
                "success": "success_passing"
            })
    
    # Merge targets, carries, and QB stats
    player_game = pd.merge(
        targets,
        carries,
        on=["game_id", "player_id", "team"],
        how="outer",
        suffixes=("", "_rush")
    )
    
    if len(qb_stats) > 0:
        player_game = pd.merge(
            player_game,
            qb_stats,
            on=["game_id", "player_id", "team"],
            how="outer"
        )
    
    # Fill missing values with 0
    for col in ["targets", "receptions", "receiving_yards", "epa_receiving", 
                "success_receiving", "carries", "rushing_yards", "epa_rushing", 
                "success_rushing", "pass_attempts", "completions", "passing_yards",
                "passing_tds", "interceptions", "epa_passing", "success_passing"]:
        if col in player_game.columns:
            player_game[col] = player_game[col].fillna(0)
    
    # Calculate totals (include QB passing EPA)
    player_game["epa_total"] = (
        player_game.get("epa_receiving", 0) + 
        player_game.get("epa_rushing", 0) + 
        player_game.get("epa_passing", 0)
    )
    
    # EPA per opportunity
    player_game["epa_per_target"] = player_game.apply(
        lambda x: x["epa_receiving"] / x["targets"] if x["targets"] > 0 else 0.0,
        axis=1
    )
    player_game["epa_per_rush"] = player_game.apply(
        lambda x: x["epa_rushing"] / x["carries"] if x["carries"] > 0 else 0.0,
        axis=1
    )
    player_game["epa_per_attempt"] = player_game.apply(
        lambda x: x["epa_passing"] / x["pass_attempts"] if x.get("pass_attempts", 0) > 0 else 0.0,
        axis=1
    )
    
    # Completion rate for QBs
    player_game["completion_rate"] = player_game.apply(
        lambda x: x["completions"] / x["pass_attempts"] if x.get("pass_attempts", 0) > 0 else 0.0,
        axis=1
    )
    
    # Success rate (successful plays / total plays)
    total_plays = player_game["targets"] + player_game["carries"]
    total_success = player_game.get("success_receiving", pd.Series([0] * len(player_game))) + player_game.get("success_rushing", pd.Series([0] * len(player_game)))
    player_game["success_rate"] = player_game.apply(
        lambda x: (x.get("success_receiving", 0) + x.get("success_rushing", 0)) / (x["targets"] + x["carries"]) 
        if (x["targets"] + x["carries"]) > 0 else 0.0,
        axis=1
    )
    
    # Redzone touches (targets + carries in redzone)
    # Get redzone plays (inside 20 yard line)
    if "yardline_100" in pass_plays.columns:
        redzone_pass_filter = (
            (pass_plays[target_id_col].notna()) &
            (pass_plays["yardline_100"] <= 20)
        )
        redzone_passes = pass_plays[redzone_pass_filter].groupby(
            ["game_id", target_id_col, "posteam"]
        ).size().reset_index(name="rz_targets")
        redzone_passes = redzone_passes.rename(columns={target_id_col: "player_id", "posteam": "team"})
    else:
        redzone_passes = pd.DataFrame(columns=["game_id", "player_id", "team", "rz_targets"])
    
    if "yardline_100" in rush_plays.columns:
        redzone_rush_filter = (
            (rush_plays[rusher_id_col].notna()) &
            (rush_plays["yardline_100"] <= 20)
        )
        redzone_rushes = rush_plays[redzone_rush_filter].groupby(
            ["game_id", rusher_id_col, "posteam"]
        ).size().reset_index(name="rz_carries")
        redzone_rushes = redzone_rushes.rename(columns={rusher_id_col: "player_id", "posteam": "team"})
    else:
        redzone_rushes = pd.DataFrame(columns=["game_id", "player_id", "team", "rz_carries"])
    
    rz_touches = pd.merge(
        redzone_passes,
        redzone_rushes,
        on=["game_id", "player_id", "team"],
        how="outer"
    ).fillna(0)
    rz_touches["redzone_touches"] = rz_touches["rz_targets"] + rz_touches["rz_carries"]
    rz_touches = rz_touches[["game_id", "player_id", "team", "redzone_touches"]]
    
    player_game = pd.merge(
        player_game,
        rz_touches,
        on=["game_id", "player_id", "team"],
        how="left"
    )
    player_game["redzone_touches"] = player_game["redzone_touches"].fillna(0).astype(int)
    
    # Add opponent (from games table)
    player_game = pd.merge(
        player_game,
        games[["game_id", "home_team", "away_team"]],
        on="game_id",
        how="left"
    )
    # Set opponent: if player's team is home, opponent is away, and vice versa
    player_game["opponent"] = player_game.apply(
        lambda x: x["away_team"] if x["team"] == x["home_team"] else x["home_team"],
        axis=1
    )
    
    # Select final columns (include QB stats if present)
    # Include epa_receiving and epa_rushing for transparency
    base_cols = [
        "game_id", "player_id", "team", "opponent",
        "targets", "receptions", "receiving_yards",
        "carries", "rushing_yards",
        "epa_receiving", "epa_rushing", "epa_total", 
        "epa_per_target", "epa_per_rush",
        "success_rate", "redzone_touches"
    ]
    
    # Add QB columns if they exist
    qb_cols = ["pass_attempts", "completions", "passing_yards", "passing_tds", 
               "interceptions", "epa_passing", "epa_per_attempt", "completion_rate"]
    final_cols = base_cols + [col for col in qb_cols if col in player_game.columns]
    
    # Only select columns that actually exist
    final_cols = [col for col in final_cols if col in player_game.columns]
    player_game = player_game[final_cols].copy()
    
    # Sort by game_id, player_id
    player_game = player_game.sort_values(["game_id", "player_id"]).reset_index(drop=True)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        player_game.to_parquet(save_path, index=False, compression="snappy")
        logger.info(f"Saved player_game_offense table to {save_path}")
    
    return player_game


def build_team_game_offense_table(
    pbp: pd.DataFrame,
    games: pd.DataFrame,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build team_game_offense table from play-by-play data.
    
    Schema: game_id, team,
            team_plays, dropbacks, rushes, pass_rate,
            team_epa_per_play
    """
    logger.info("Building team_game_offense table")
    
    # Filter to offensive plays with EPA
    offense_plays = pbp[
        (pbp["epa"].notna()) &
        (pbp["play_type"].isin(["pass", "run"])) &
        (pbp["posteam"].notna())
    ].copy()
    
    # Aggregate by game and team
    team_stats = offense_plays.groupby(["game_id", "posteam"]).agg({
        "play_id": "count",  # Total plays
        "epa": "sum",
        "play_type": lambda x: (x == "pass").sum(),  # Dropbacks (pass plays)
    }).reset_index()
    
    team_stats = team_stats.rename(columns={
        "posteam": "team",
        "play_id": "team_plays",
        "play_type": "dropbacks"
    })
    
    # Calculate rushes
    team_stats["rushes"] = team_stats["team_plays"] - team_stats["dropbacks"]
    
    # Pass rate
    team_stats["pass_rate"] = team_stats.apply(
        lambda x: x["dropbacks"] / x["team_plays"] if x["team_plays"] > 0 else 0.0,
        axis=1
    )
    
    # EPA per play
    team_stats["team_epa_per_play"] = team_stats.apply(
        lambda x: x["epa"] / x["team_plays"] if x["team_plays"] > 0 else 0.0,
        axis=1
    )
    
    # Select final columns
    team_stats = team_stats[[
        "game_id", "team",
        "team_plays", "dropbacks", "rushes", "pass_rate",
        "team_epa_per_play"
    ]].copy()
    
    # Sort by game_id, team
    team_stats = team_stats.sort_values(["game_id", "team"]).reset_index(drop=True)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        team_stats.to_parquet(save_path, index=False, compression="snappy")
        logger.info(f"Saved team_game_offense table to {save_path}")
    
    return team_stats


def build_player_game_defense_table(
    pbp: pd.DataFrame,
    rosters: pd.DataFrame,
    games: pd.DataFrame,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build player_game_defense table from play-by-play data.
    
    Schema: game_id, player_id, team, opponent,
            tackles, solo_tackles, assist_tackles,
            sacks, qb_hits, interceptions, passes_defended,
            fumbles_forced, fumbles_recovered,
            epa_allowed (negative EPA generated by offense against this player's defense)
    """
    logger.info("Building player_game_defense table")
    
    # Filter to plays with defensive stats
    defense_plays = pbp[
        (pbp["epa"].notna()) &
        (pbp["play_type"].isin(["pass", "run"])) &
        (pbp["defteam"].notna())
    ].copy()
    
    player_def_stats = []
    
    # Find defensive player ID columns
    sack_id_col = None
    int_id_col = None
    tackle_id_cols = []
    
    for col in ["sack_player_id", "sack_player", "sacker_id"]:
        if col in defense_plays.columns:
            sack_id_col = col
            break
    
    for col in ["interception_player_id", "interception_player", "interceptor_id"]:
        if col in defense_plays.columns:
            int_id_col = col
            break
    
    # Tackles - can be solo or assist
    for col in ["tackle_player_id", "tackle_player", "tackler_id", "solo_tackle_player_id", "assist_tackle_player_id"]:
        if col in defense_plays.columns:
            tackle_id_cols.append(col)
    
    # Sacks
    sack_stats = pd.DataFrame()
    if sack_id_col and sack_id_col in defense_plays.columns:
        sack_plays = defense_plays[defense_plays["sack"] == 1].copy()
        if len(sack_plays) > 0:
            sack_stats = sack_plays.groupby(
                ["game_id", sack_id_col, "defteam"]
            ).agg({
                "sack": "sum",
                "qb_hit": "sum"  # QB hits often accompany sacks
            }).reset_index()
            sack_stats = sack_stats.rename(columns={
                sack_id_col: "player_id",
                "defteam": "team",
                "sack": "sacks"
            })
    
    # Interceptions
    int_stats = pd.DataFrame()
    if int_id_col and int_id_col in defense_plays.columns:
        int_plays = defense_plays[defense_plays["interception"] == 1].copy()
        if len(int_plays) > 0:
            int_stats = int_plays.groupby(
                ["game_id", int_id_col, "defteam"]
            ).agg({
                "interception": "sum"
            }).reset_index()
            int_stats = int_stats.rename(columns={
                int_id_col: "player_id",
                "defteam": "team",
                "interception": "interceptions"
            })
    
    # Tackles (solo and assist)
    tackle_stats = pd.DataFrame()
    if len(tackle_id_cols) > 0:
        tackle_data = []
        for col in tackle_id_cols:
            if col in defense_plays.columns:
                tackle_plays = defense_plays[defense_plays[col].notna()].copy()
                if len(tackle_plays) > 0:
                    # Determine if solo or assist
                    is_solo = "solo" in col.lower()
                    tackle_type = "solo_tackles" if is_solo else "assist_tackles"
                    
                    tackle_group = tackle_plays.groupby(
                        ["game_id", col, "defteam"]
                    ).size().reset_index(name=tackle_type)
                    tackle_group = tackle_group.rename(columns={col: "player_id", "defteam": "team"})
                    tackle_data.append(tackle_group)
        
        if len(tackle_data) > 0:
            tackle_stats = tackle_data[0]
            for df in tackle_data[1:]:
                tackle_stats = pd.merge(
                    tackle_stats, df,
                    on=["game_id", "player_id", "team"],
                    how="outer",
                    suffixes=("", "_new")
                )
                # Merge duplicate columns
                for col in df.columns:
                    if col.endswith("_new"):
                        base_col = col.replace("_new", "")
                        if base_col in tackle_stats.columns:
                            tackle_stats[base_col] = tackle_stats[base_col].fillna(0) + tackle_stats[col].fillna(0)
                            tackle_stats = tackle_stats.drop(columns=[col])
    
    # QB hits (separate from sacks)
    qb_hit_stats = pd.DataFrame()
    if "qb_hit_player_id" in defense_plays.columns:
        qb_hit_plays = defense_plays[defense_plays["qb_hit"] == 1].copy()
        if len(qb_hit_plays) > 0:
            qb_hit_stats = qb_hit_plays.groupby(
                ["game_id", "qb_hit_player_id", "defteam"]
            ).agg({
                "qb_hit": "sum"
            }).reset_index()
            qb_hit_stats = qb_hit_stats.rename(columns={
                "qb_hit_player_id": "player_id",
                "defteam": "team",
                "qb_hit": "qb_hits"
            })
    
    # Fumbles forced and recovered
    fumble_stats = pd.DataFrame()
    if "fumble_forced_player_id" in defense_plays.columns:
        fumble_forced = defense_plays[defense_plays["fumble_forced"] == 1].copy()
        if len(fumble_forced) > 0:
            fumble_forced_stats = fumble_forced.groupby(
                ["game_id", "fumble_forced_player_id", "defteam"]
            ).agg({
                "fumble_forced": "sum"
            }).reset_index()
            fumble_forced_stats = fumble_forced_stats.rename(columns={
                "fumble_forced_player_id": "player_id",
                "defteam": "team",
                "fumble_forced": "fumbles_forced"
            })
            fumble_stats = fumble_forced_stats
    
    # Merge all defensive stats
    player_def = pd.DataFrame()
    
    # Start with tackles (most common)
    if len(tackle_stats) > 0:
        player_def = tackle_stats.copy()
    else:
        # Create empty DataFrame with required columns
        player_def = pd.DataFrame(columns=["game_id", "player_id", "team"])
    
    # Merge sacks
    if len(sack_stats) > 0:
        player_def = pd.merge(
            player_def, sack_stats[["game_id", "player_id", "team", "sacks"]],
            on=["game_id", "player_id", "team"],
            how="outer"
        )
    
    # Merge interceptions
    if len(int_stats) > 0:
        player_def = pd.merge(
            player_def, int_stats[["game_id", "player_id", "team", "interceptions"]],
            on=["game_id", "player_id", "team"],
            how="outer"
        )
    
    # Merge QB hits
    if len(qb_hit_stats) > 0:
        player_def = pd.merge(
            player_def, qb_hit_stats[["game_id", "player_id", "team", "qb_hits"]],
            on=["game_id", "player_id", "team"],
            how="outer"
        )
    
    # Merge fumbles
    if len(fumble_stats) > 0:
        player_def = pd.merge(
            player_def, fumble_stats[["game_id", "player_id", "team", "fumbles_forced"]],
            on=["game_id", "player_id", "team"],
            how="outer"
        )
    
    # Fill missing values with 0
    for col in ["solo_tackles", "assist_tackles", "sacks", "qb_hits", "interceptions", "fumbles_forced"]:
        if col in player_def.columns:
            player_def[col] = player_def[col].fillna(0)
        else:
            player_def[col] = 0
    
    # Calculate total tackles
    player_def["tackles"] = (
        player_def.get("solo_tackles", 0) + 
        player_def.get("assist_tackles", 0)
    )
    
    # Calculate EPA allowed (negative of offensive EPA when this player's team is on defense)
    # EPA allowed = -EPA generated by offense
    if len(player_def) > 0:
        # Get EPA for plays where this player's team was on defense
        def_epa = defense_plays.groupby(["game_id", "defteam"]).agg({
            "epa": "sum"  # This is offensive EPA, so negative = good defense
        }).reset_index()
        def_epa = def_epa.rename(columns={"defteam": "team", "epa": "epa_allowed"})
        def_epa["epa_allowed"] = -def_epa["epa_allowed"]  # Negative because lower offensive EPA = better defense
        
        # Merge with player stats (each player gets team's EPA allowed)
        player_def = pd.merge(
            player_def, def_epa,
            on=["game_id", "team"],
            how="left"
        )
        player_def["epa_allowed"] = player_def["epa_allowed"].fillna(0)
    
    # Merge with games to get opponent
    player_def = pd.merge(
        player_def,
        games[["game_id", "home_team", "away_team"]],
        on="game_id",
        how="left"
    )
    player_def["opponent"] = player_def.apply(
        lambda x: x["away_team"] if x["team"] == x["home_team"] else x["home_team"],
        axis=1
    )
    
    # Select final columns
    final_cols = [
        "game_id", "player_id", "team", "opponent",
        "tackles", "solo_tackles", "assist_tackles",
        "sacks", "qb_hits", "interceptions", "fumbles_forced",
        "epa_allowed"
    ]
    final_cols = [col for col in final_cols if col in player_def.columns]
    player_def = player_def[final_cols].copy()
    
    # Sort by game_id, player_id
    player_def = player_def.sort_values(["game_id", "player_id"]).reset_index(drop=True)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        player_def.to_parquet(save_path, index=False, compression="snappy")
        logger.info(f"Saved player_game_defense table to {save_path}")
    
    return player_def


def build_team_game_defense_table(
    pbp: pd.DataFrame,
    games: pd.DataFrame,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build team_game_defense table from play-by-play data.
    
    Schema: game_id, team,
            plays_faced, dropbacks_faced, rushes_faced,
            epa_allowed_per_play, sacks, interceptions, fumbles_forced
    """
    logger.info("Building team_game_defense table")
    
    # Filter to plays where team was on defense
    defense_plays = pbp[
        (pbp["epa"].notna()) &
        (pbp["play_type"].isin(["pass", "run"])) &
        (pbp["defteam"].notna())
    ].copy()
    
    # Aggregate by game and defensive team
    team_def = defense_plays.groupby(["game_id", "defteam"]).agg({
        "play_id": "count",  # Total plays faced
        "play_type": lambda x: (x == "pass").sum(),  # Dropbacks faced
        "epa": "sum",  # Offensive EPA (negative = good defense)
        "sack": "sum",
        "interception": "sum",
        "fumble_forced": "sum"
    }).reset_index()
    
    team_def = team_def.rename(columns={
        "defteam": "team",
        "play_id": "plays_faced",
        "play_type": "dropbacks_faced",
        "epa": "epa_allowed",
        "sack": "sacks",
        "interception": "interceptions",
        "fumble_forced": "fumbles_forced"
    })
    
    # Calculate rushes faced
    team_def["rushes_faced"] = team_def["plays_faced"] - team_def["dropbacks_faced"]
    
    # EPA allowed per play (negative because lower offensive EPA = better defense)
    team_def["epa_allowed"] = -team_def["epa_allowed"]
    team_def["epa_allowed_per_play"] = team_def.apply(
        lambda x: x["epa_allowed"] / x["plays_faced"] if x["plays_faced"] > 0 else 0.0,
        axis=1
    )
    
    # Select final columns
    team_def = team_def[[
        "game_id", "team",
        "plays_faced", "dropbacks_faced", "rushes_faced",
        "epa_allowed", "epa_allowed_per_play",
        "sacks", "interceptions", "fumbles_forced"
    ]].copy()
    
    # Sort by game_id, team
    team_def = team_def.sort_values(["game_id", "team"]).reset_index(drop=True)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        team_def.to_parquet(save_path, index=False, compression="snappy")
        logger.info(f"Saved team_game_defense table to {save_path}")
    
    return team_def


def build_all_tables(
    pbp_path: Optional[Path] = None,
    rosters_path: Optional[Path] = None,
    schedules_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
):
    """
    Build all core tables from ingested data.
    
    Args:
        pbp_path: Path to play-by-play Parquet file
        rosters_path: Path to rosters Parquet file
        schedules_path: Path to schedules Parquet file
        output_dir: Directory to save output tables
    """
    output_dir = output_dir or PARQUET_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading ingested data")
    pbp = pd.read_parquet(pbp_path or output_dir / "pbp.parquet")
    rosters = pd.read_parquet(rosters_path or output_dir / "rosters.parquet")
    schedules = pd.read_parquet(schedules_path or output_dir / "schedules.parquet")
    
    # Build tables in order
    games = build_games_table(schedules, output_dir / "games.parquet")
    players = build_players_table(rosters, output_dir / "players.parquet")
    player_game_offense = build_player_game_offense_table(
        pbp, rosters, games, output_dir / "player_game_offense.parquet"
    )
    team_game_offense = build_team_game_offense_table(
        pbp, games, output_dir / "team_game_offense.parquet"
    )
    
    # Build defense tables
    player_game_defense = build_player_game_defense_table(
        pbp, rosters, games, output_dir / "player_game_defense.parquet"
    )
    team_game_defense = build_team_game_defense_table(
        pbp, games, output_dir / "team_game_defense.parquet"
    )
    
    logger.info("All tables built successfully")
    
    return {
        "games": games,
        "players": players,
        "player_game_offense": player_game_offense,
        "team_game_offense": team_game_offense,
        "player_game_defense": player_game_defense,
        "team_game_defense": team_game_defense
    }
